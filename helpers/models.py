import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from helpers.basic import *

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out





class inceptionBlock_light(nn.Module):
    def __init__(self,in_channels, ch1x1,ch3x3,ch5x5,ch3x3_in,ch5x5_in,pool_proj=32) -> None:
        super(inceptionBlock_light,self).__init__()
        self.branch1=conv3x3_bn_relu(in_channels,ch1x1,kernel_size=1,padding=0)
        self.branch2=nn.Sequential(conv3x3_bn_relu(in_channels,ch3x3_in,kernel_size=1,padding=0),conv3x3_bn_relu(ch3x3_in,ch3x3,kernel_size=3))
        self.branch3=nn.Sequential(conv3x3_bn_relu(in_channels,ch5x5_in,kernel_size=1,padding=0),conv3x3_bn_relu(ch5x5_in,ch5x5,kernel_size=5,padding=2))
        #self.branch4=nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),conv3x3_bn_relu(in_channels,pool_proj,kernel_size=1,padding=0))
    def forward(self, input):
        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs,1)

class Inception(nn.Module) :
    def __init__(self, in_dim, out_dim1, mid_dim3, out_dim3, mid_dim5, out_dim5, pool):
        super(Inception, self).__init__()
        self.lay1 = nn.Sequential(nn.Conv2d(in_dim, out_dim1, kernel_size= 1), nn.BatchNorm2d(out_dim1), nn.ReLU())
        self.lay2 = nn.Sequential(nn.Conv2d(in_dim, mid_dim3, kernel_size = 1), nn.BatchNorm2d(mid_dim3), nn.ReLU(), nn.Conv2d(mid_dim3, out_dim3, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_dim3), nn.ReLU())
        self.lay3 = nn.Sequential(nn.Conv2d(in_dim, mid_dim5, kernel_size = 1), nn.BatchNorm2d(mid_dim5), nn.ReLU(), nn.Conv2d(mid_dim5, out_dim5, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_dim5), nn.ReLU(), nn.Conv2d(out_dim5, out_dim5, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_dim5), nn.ReLU())
        self.lay4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_dim, pool, kernel_size = 1), nn.BatchNorm2d(pool), nn.ReLU())

    def forward(self, x):
        y1 = self.lay1(x)
        y2 = self.lay2(x)
        y3 = self.lay3(x)
        y4 = self.lay4(x)

        return torch.cat([y1, y2, y3, y4], 1)

class Inception_NoBatch(nn.Module) :
    def __init__(self, in_dim, out_dim1, mid_dim3, out_dim3, mid_dim5, out_dim5, pool):
        super(Inception_NoBatch, self).__init__()
        self.lay1 = nn.Sequential(nn.Conv2d(in_dim, out_dim1, kernel_size= 1), nn.ReLU())
        self.lay2 = nn.Sequential(nn.Conv2d(in_dim, mid_dim3, kernel_size = 1), nn.ReLU(), nn.Conv2d(mid_dim3, out_dim3, kernel_size = 3, padding = 1), nn.ReLU())
        self.lay3 = nn.Sequential(nn.Conv2d(in_dim, mid_dim5, kernel_size = 1), nn.ReLU(), nn.Conv2d(mid_dim5, out_dim5, kernel_size = 3, padding = 1), nn.ReLU(), nn.Conv2d(out_dim5, out_dim5, kernel_size = 3, padding = 1), nn.ReLU())
        self.lay4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_dim, pool, kernel_size = 1), nn.ReLU())

    def forward(self, x):
        y1 = self.lay1(x)
        y2 = self.lay2(x)
        y3 = self.lay3(x)
        y4 = self.lay4(x)

        return torch.cat([y1, y2, y3, y4], 1)





class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.enc_2=ResNetBlock(in_channels=64, out_channels=64, stride=2)
        self.enc_3=ResNetBlock(in_channels=64, out_channels=128, stride=2)
        self.enc_4=ResNetBlock(in_channels=128, out_channels=128, stride=1)
        self.enc_5=ResNetBlock(in_channels=128, out_channels=256, stride=1)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=256,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,input):

        rgb = input['rgb']
        d = input['d']

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=4,kernel_size=5,out_channels=32,stride=2, padding=2)

        #Encoder
        self.enc_1=CBAM(32)
        self.conv_intermediate=conv3x3_relu(32,64,stride=2)
        self.enc_2=CBAM(64)
        self.conv_intermediate_2=conv3x3_relu(64,128,stride=2)
        self.enc_3=CBAM(128)
        self.enc_4=CBAM(128)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,input):

        rgb = input['rgb']
        d = input['d']

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        out=self.conv_intermediate(encoder_feature_1)
        encoder_feature_2=self.enc_2(out)
        out=self.conv_intermediate_2(encoder_feature_2)
        encoder_feature_3=self.enc_3(out)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        
        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_4)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

        #Encoder
        self.enc_1=conv3x3_relu(in_channels=32, out_channels=64, stride=2)
        self.enc_2=conv3x3_relu(in_channels=64, out_channels=64, stride=2)
        self.enc_3=conv3x3_relu(in_channels=64, out_channels=128, stride=2)
        self.enc_4=conv3x3_relu(in_channels=128, out_channels=128, stride=1)
        self.enc_5=conv3x3_relu(in_channels=128, out_channels=256, stride=1)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=256,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,input):

        rgb = input['rgb']
        d = input['d']

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class InceptionModel(nn.Module):
    def __init__(self):
        super(InceptionModel, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=4,kernel_size=5,out_channels=48,stride=2, padding=2)

        #Encoder
        #self.enc_1=conv3x3_relu(in_channels=32, out_channels=64, stride=1)
        #self.enc_2=conv3x3_relu(in_channels=64, out_channels=128, stride=1)
        self.inception_1=Inception_NoBatch(48, 16, 24, 32, 4, 8, 8)
        self.intermediate_conv=conv3x3_relu(in_channels=64, out_channels=64, stride=2)
        self.inception_2=Inception_NoBatch(64, 32, 32, 48, 8, 24, 8)
        self.intermediate_conv_2=conv3x3_relu(in_channels=112, out_channels=120, stride=2) 
        self.inception_3=Inception_NoBatch(120, 48, 24, 52, 4, 12, 16)
        self.inception_4=Inception_NoBatch(128, 40, 28, 56, 6, 16, 16)
        #self.avgpool = nn.AvgPool2d(512, stride = 1)
        #self.inception_3=Inception_NoBatch() #256
        #self.inception_4=Inception_NoBatch() #512

        #Decoder
        #self.dec_1=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=100,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=100, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,input):

        rgb = input['rgb']
        d = input['d']

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        #encoder_feature_1=self.enc_1(encoder_feature_init)
        #encoder_feature_2=self.enc_2(encoder_feature_1)
        # encoder_feature_3=self.enc_3(encoder_feature_2)
        # encoder_feature_4=self.enc_4(encoder_feature_3)
        # encoder_feature_5=self.enc_5(encoder_feature_4)

        out=self.inception_1(encoder_feature_init)        
        #print("Shapes->"+str(out.shape)) 
        out=self.intermediate_conv(out)   
        out=self.inception_2(out)
        #print("Shapes->"+str(out.shape))
        out=self.intermediate_conv_2(out)   
        out=self.inception_3(out)
        #print("Shapes->"+str(out.shape))
        out=self.inception_4(out)
        #print("Shapes->"+str(out.shape))


        #Decoder
        #decoder_feature_1=self.dec_1(out)
        #decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        out=self.dec_2(out)
        #decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(out)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class InceptionModel_LateFusion(nn.Module):
    def __init__(self):
        super(InceptionModel_LateFusion, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=1,kernel_size=5,out_channels=48,stride=2, padding=2)
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=5,out_channels=8,stride=4, padding=2)

        #Encoder
        #self.enc_1=conv3x3_relu(in_channels=32, out_channels=64, stride=1)
        #self.enc_2=conv3x3_relu(in_channels=64, out_channels=128, stride=1)
        self.inception_1=Inception_NoBatch(48, 16, 24, 32, 4, 8, 8)
        self.intermediate_conv=conv3x3_relu(in_channels=64, out_channels=64, stride=2)
        self.inception_2=Inception_NoBatch(64, 32, 32, 48, 8, 24, 2)#106
        self.intermediate_conv_2=conv3x3_relu(in_channels=114, out_channels=120, stride=2) 
        self.inception_3=Inception_NoBatch(120, 48, 24, 52, 4, 12, 16)
        self.inception_4=Inception_NoBatch(128, 40, 28, 56, 6, 16, 16)
        #self.avgpool = nn.AvgPool2d(512, stride = 1)
        #self.inception_3=Inception_NoBatch() #256
        #self.inception_4=Inception_NoBatch() #512

        #Decoder
        #self.dec_1=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=100,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=100, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,input):

        rgb = input['rgb']
        d = input['d']

        #join the rgb and the sparse information
        #encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        encoder_feature_init=self.first_layer(d)
        rgb_out=self.first_layer_rgb(rgb)
        #print(encoder_feature_init.shape)
        #Encoder
        #encoder_feature_1=self.enc_1(encoder_feature_init)
        #encoder_feature_2=self.enc_2(encoder_feature_1)
        # encoder_feature_3=self.enc_3(encoder_feature_2)
        # encoder_feature_4=self.enc_4(encoder_feature_3)
        # encoder_feature_5=self.enc_5(encoder_feature_4)

        out=self.inception_1(encoder_feature_init)        
        #print("Shapes->"+str(out.shape)) 
        out=self.intermediate_conv(out)   
        out=self.inception_2(out)
        #print("Shapes->"+str(out.shape))
        out=self.intermediate_conv_2(torch.cat((rgb_out, out),dim=1))   
        out=self.inception_3(out)
        #print("Shapes->"+str(out.shape))
        out=self.inception_4(out)
        #print("Shapes->"+str(out.shape))


        #Decoder
        #decoder_feature_1=self.dec_1(out)
        #decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        out=self.dec_2(out)
        #decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(out)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 


class CNNModel_LateFusion(nn.Module):
    def __init__(self):
        super(CNNModel_LateFusion, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=1,kernel_size=5,out_channels=32,stride=1, padding=2)
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=5,out_channels=16,stride=4, padding=2)

        #Encoder
        self.enc_1=conv3x3_relu(in_channels=32, out_channels=64, stride=2)
        self.enc_2=conv3x3_relu(in_channels=64, out_channels=64, stride=2)
        self.enc_3=conv3x3_relu(in_channels=80, out_channels=128, stride=2)
        self.enc_4=conv3x3_relu(in_channels=128, out_channels=128, stride=1)
        self.enc_5=conv3x3_relu(in_channels=128, out_channels=256, stride=1)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=256,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,input):

        rgb = input['rgb']
        d = input['d']

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(d)
        rgb_out=self.first_layer_rgb(rgb)
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(torch.cat((rgb_out, encoder_feature_2),dim=1))
        encoder_feature_4=self.enc_4(encoder_feature_3)
        encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 


########################### MODELS FOR TENSORRT ###############################

class CNNModel_LateFusion_RT(nn.Module):
    def __init__(self):
        super(CNNModel_LateFusion_RT, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=1,kernel_size=5,out_channels=32,stride=1, padding=2)
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=5,out_channels=16,stride=4, padding=2)

        #Encoder
        self.enc_1=conv3x3_relu(in_channels=32, out_channels=64, stride=2)
        self.enc_2=conv3x3_relu(in_channels=64, out_channels=64, stride=2)
        self.enc_3=conv3x3_relu(in_channels=80, out_channels=128, stride=2)
        self.enc_4=conv3x3_relu(in_channels=128, out_channels=128, stride=1)
        self.enc_5=conv3x3_relu(in_channels=128, out_channels=256, stride=1)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=256,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,*input):
        
        rgb= input[0]
        d=input[1]

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(d)
        rgb_out=self.first_layer_rgb(rgb)
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(torch.cat((rgb_out, encoder_feature_2),dim=1))
        encoder_feature_4=self.enc_4(encoder_feature_3)
        encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class ResNetModel_RT(nn.Module):
    def __init__(self):
        super(ResNetModel_RT, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.enc_2=ResNetBlock(in_channels=64, out_channels=64, stride=2)
        self.enc_3=ResNetBlock(in_channels=64, out_channels=128, stride=2)
        self.enc_4=ResNetBlock(in_channels=128, out_channels=128, stride=1)
        self.enc_5=ResNetBlock(in_channels=128, out_channels=256, stride=1)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=256,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,*input):

        rgb = input[0]
        d = input[1]

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class ResNetModel_LateFusion_RT(nn.Module):
    def __init__(self):
        super(ResNetModel_LateFusion_RT, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=1,kernel_size=5,out_channels=32,stride=1, padding=2)
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=5,out_channels=16,stride=4, padding=2)

        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.enc_2=ResNetBlock(in_channels=64, out_channels=64, stride=2)
        self.enc_3=ResNetBlock(in_channels=80, out_channels=128, stride=2)
        self.enc_4=ResNetBlock(in_channels=128, out_channels=128, stride=1)
        self.enc_5=ResNetBlock(in_channels=128, out_channels=256, stride=1)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=256,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,*input):

        rgb = input[0]
        d = input[1]

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(d)
        rgb_out=self.first_layer_rgb(rgb)

        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(torch.cat((rgb_out, encoder_feature_2),dim=1))
        encoder_feature_4=self.enc_4(encoder_feature_3)
        encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 


class AttentionModel_RT(nn.Module):
    def __init__(self):
        super(AttentionModel_RT, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=4,kernel_size=5,out_channels=32,stride=2, padding=2)

        #Encoder
        self.enc_1=CBAM(32)
        self.conv_intermediate=conv3x3_relu(32,64,stride=2)
        self.enc_2=CBAM(64)
        self.conv_intermediate_2=conv3x3_relu(64,128,stride=2)
        self.enc_3=CBAM(128)
        self.enc_4=CBAM(128)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,*input):

        rgb = input[0]
        d = input[1]

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        out=self.conv_intermediate(encoder_feature_1)
        encoder_feature_2=self.enc_2(out)
        out=self.conv_intermediate_2(encoder_feature_2)
        encoder_feature_3=self.enc_3(out)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        
        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_4)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class AttentionModel_LateFusion_RT(nn.Module):
    def __init__(self):
        super(AttentionModel_LateFusion_RT, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=1,kernel_size=5,out_channels=32,stride=2, padding=2)
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=5,out_channels=16,stride=4, padding=2)
        #Encoder
        self.enc_1=CBAM(32)
        self.conv_intermediate=conv3x3_relu(32,64,stride=2)
        self.enc_2=CBAM(64)
        self.conv_intermediate_2=conv3x3_relu(80,128,stride=2)
        self.enc_3=CBAM(128)
        self.enc_4=CBAM(128)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,*input):

        rgb = input[0]
        d = input[1]

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(d)
        rgb_out=self.first_layer_rgb(rgb)
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        out=self.conv_intermediate(encoder_feature_1)
        encoder_feature_2=self.enc_2(out)
        out=self.conv_intermediate_2(torch.cat((rgb_out, encoder_feature_2),dim=1))
        encoder_feature_3=self.enc_3(out)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        
        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_4)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class CNNModel_RT(nn.Module):
    def __init__(self):
        super(CNNModel_RT, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

        #Encoder
        self.enc_1=conv3x3_relu(in_channels=32, out_channels=64, stride=2)
        self.enc_2=conv3x3_relu(in_channels=64, out_channels=64, stride=2)
        self.enc_3=conv3x3_relu(in_channels=64, out_channels=128, stride=2)
        self.enc_4=conv3x3_relu(in_channels=128, out_channels=128, stride=1)
        self.enc_5=conv3x3_relu(in_channels=128, out_channels=256, stride=1)
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=256,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=1)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,*input):

        rgb = input[0]
        d = input[1]

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class InceptionModel_RT(nn.Module):
    def __init__(self):
        super(InceptionModel_RT, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=4,kernel_size=5,out_channels=48,stride=2, padding=2)

        #Encoder
        #self.enc_1=conv3x3_relu(in_channels=32, out_channels=64, stride=1)
        #self.enc_2=conv3x3_relu(in_channels=64, out_channels=128, stride=1)
        self.inception_1=Inception_NoBatch(48, 16, 24, 32, 4, 8, 8)
        self.intermediate_conv=conv3x3_relu(in_channels=64, out_channels=64, stride=2)
        self.inception_2=Inception_NoBatch(64, 32, 32, 48, 8, 24, 8)
        self.intermediate_conv_2=conv3x3_relu(in_channels=112, out_channels=120, stride=2) 
        self.inception_3=Inception_NoBatch(120, 48, 24, 52, 4, 12, 16)
        self.inception_4=Inception_NoBatch(128, 40, 28, 56, 6, 16, 16)
        #self.avgpool = nn.AvgPool2d(512, stride = 1)
        #self.inception_3=Inception_NoBatch() #256
        #self.inception_4=Inception_NoBatch() #512

        #Decoder
        #self.dec_1=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=100,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=100, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,*input):

        rgb = input[0]
        d = input[1]

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        #encoder_feature_1=self.enc_1(encoder_feature_init)
        #encoder_feature_2=self.enc_2(encoder_feature_1)
        # encoder_feature_3=self.enc_3(encoder_feature_2)
        # encoder_feature_4=self.enc_4(encoder_feature_3)
        # encoder_feature_5=self.enc_5(encoder_feature_4)

        out=self.inception_1(encoder_feature_init)        
        #print("Shapes->"+str(out.shape)) 
        out=self.intermediate_conv(out)   
        out=self.inception_2(out)
        #print("Shapes->"+str(out.shape))
        out=self.intermediate_conv_2(out)   
        out=self.inception_3(out)
        #print("Shapes->"+str(out.shape))
        out=self.inception_4(out)
        #print("Shapes->"+str(out.shape))


        #Decoder
        #decoder_feature_1=self.dec_1(out)
        #decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        out=self.dec_2(out)
        #decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(out)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 

class InceptionModel_LateFusion_RT(nn.Module):
    def __init__(self):
        super(InceptionModel_LateFusion_RT, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_relu(in_channels=1,kernel_size=5,out_channels=48,stride=2, padding=2)
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=5,out_channels=8,stride=4, padding=2)

        #Encoder
        #self.enc_1=conv3x3_relu(in_channels=32, out_channels=64, stride=1)
        #self.enc_2=conv3x3_relu(in_channels=64, out_channels=128, stride=1)
        self.inception_1=Inception_NoBatch(48, 16, 24, 32, 4, 8, 8)
        self.intermediate_conv=conv3x3_relu(in_channels=64, out_channels=64, stride=2)
        self.inception_2=Inception_NoBatch(64, 32, 32, 48, 8, 24, 2)#106
        self.intermediate_conv_2=conv3x3_relu(in_channels=114, out_channels=120, stride=2) 
        self.inception_3=Inception_NoBatch(120, 48, 24, 52, 4, 12, 16)
        self.inception_4=Inception_NoBatch(128, 40, 28, 56, 6, 16, 16)
        #self.avgpool = nn.AvgPool2d(512, stride = 1)
        #self.inception_3=Inception_NoBatch() #256
        #self.inception_4=Inception_NoBatch() #512

        #Decoder
        #self.dec_1=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=100,padding=1,scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=100, out_channels=64,padding=1,scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        #self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1,scale_factor=2)
        self.dec_6=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)
        self.final_sigmoid=nn.Sigmoid()

        init_weights(self)
    def forward(self,*input):

        rgb = input[0]
        d = input[1]

        #join the rgb and the sparse information
        #encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        encoder_feature_init=self.first_layer(d)
        rgb_out=self.first_layer_rgb(rgb)
        #print(encoder_feature_init.shape)
        #Encoder
        #encoder_feature_1=self.enc_1(encoder_feature_init)
        #encoder_feature_2=self.enc_2(encoder_feature_1)
        # encoder_feature_3=self.enc_3(encoder_feature_2)
        # encoder_feature_4=self.enc_4(encoder_feature_3)
        # encoder_feature_5=self.enc_5(encoder_feature_4)

        out=self.inception_1(encoder_feature_init)        
        #print("Shapes->"+str(out.shape)) 
        out=self.intermediate_conv(out)   
        out=self.inception_2(out)
        #print("Shapes->"+str(out.shape))
        out=self.intermediate_conv_2(torch.cat((rgb_out, out),dim=1))   
        out=self.inception_3(out)
        #print("Shapes->"+str(out.shape))
        out=self.inception_4(out)
        #print("Shapes->"+str(out.shape))


        #Decoder
        #decoder_feature_1=self.dec_1(out)
        #decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        out=self.dec_2(out)
        #decoder_feature_2_plus=decoder_feature_2#+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(out)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_2 #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence 
