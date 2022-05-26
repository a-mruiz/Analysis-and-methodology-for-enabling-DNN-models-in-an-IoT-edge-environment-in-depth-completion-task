import torch
from torch2trt import torch2trt
import helpers.models as models
import helpers.helper as helper
from threading import Thread
from os.path import exists
from torch2trt import TRTModule

cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))


model_name="CNNModel_LateFusion"
resnet=models.ResNetModel_RT().to(device)
resnet_late=models.ResNetModel_LateFusion_RT().to(device)
inception=models.InceptionModel_RT().to(device)
inception_late=models.InceptionModel_LateFusion_RT().to(device)
cnn=models.CNNModel_RT().to(device)
cnn_late=models.CNNModel_LateFusion_RT().to(device)
attention=models.AttentionModel_RT().to(device)
attention_late=models.AttentionModel_LateFusion_RT().to(device)

#2- Load Weights
#model.half()
#model.load_state_dict(torch.load("weights/CNNModel_LateFusion.pt", map_location=device))

#model.cuda().half()

#3- Begin tests
model_name_list=["ResNetModel","ResNetModel_LateFusion","InceptionModel","InceptionModel_LateFusion","CNNModel","CNNModel_LateFusion","AttentionModel","AttentionModel_LateFusion"]
model_list=[resnet,resnet_late,inception, inception_late,cnn,cnn_late,attention,attention_late]

#model_name_list=["AttentionModel"]
#model_list=[attention]
#Creating example inputs
example_rgb=torch.ones((1, 3, 1024, 1024)).to(device)
example_depth=torch.ones((1, 1, 1024, 1024)).to(device)
example_input=[example_rgb,example_depth]

"""
This loop will->
    -Convert the model to TensorRT FP32
    -Convert the model to TensorRT FP16
    (-Convert the model to TensorRT INT8)
    -Measure and report inference times, PSNR and MSE for:
        -PyTorch model FP32
        -PyTorch model FP16
        -TensorRT FP32 model
        -TensorRT FP16 model
        (-TensorRT INT8 model)
    -Save all the aforehead models
"""




for model_name,model in zip(model_name_list,model_list):

    print("\n####################################################")
    print("#\t\tProcesing "+str(model_name)+"                 #")
    print("####################################################")
    #1-Load weights
    model.load_state_dict(torch.load("weights/"+str(model_name)+".pt", map_location=device))

    #2-Convert to TensorRT
    try:
        #if not exists("weightsRT/"+model_name+'.pth'):
        print(f"Converting {model_name} to TensorRT FP32...")
        model_rt=torch2trt(model,example_input)
        torch.save(model_rt.state_dict(), "weightsRT/"+model_name+'.pth')
        #else:
        #    print(f"File {model_name} in TensorRT FP32 already exists, skipping")
        #    model_rt = TRTModule()
        #    model_rt.load_state_dict(torch.load("weightsRT/"+model_name+'.pth'))
        
        
        #if not exists("weightsRT/"+model_name+'_half.pth'):
        print(f"Converting {model_name} to TensorRT FP16...")
        model_rt_half=torch2trt(model,example_input, fp16_mode=True)
        torch.save(model_rt_half.state_dict(), "weightsRT/"+model_name+'_half.pth')
        #else:
        #    print(f"File {model_name} in TensorRT FP16 already exists, skipping")
        #    model_rt_half = TRTModule()
        #    model_rt_half.load_state_dict(torch.load("weightsRT/"+model_name+'_half.pth'))
        
        #if not exists("weightsRT/"+model_name+'_8bit.pth'):
        print(f"Converting {model_name} to TensorRT INT8...")
        model_rt_8bit=torch2trt(model,example_input, int8_mode=True,fp16_mode=True)
        torch.save(model_rt_8bit.state_dict(), "weightsRT/"+model_name+'_8bit.pth')
        # else:
        #     print(f"File {model_name} in TensorRT INT8 already exists, skipping")
        #     model_rt_8bit = TRTModule()
        #     model_rt_8bit.load_state_dict(torch.load("weightsRT/"+model_name+'_8bit.pth'))
    except Exception as e:
        print("\n ######################## EXCEPTION OCURRED ##################################")
        print(e)
        print("################################### END EXCEPTION ###############################\n")
    #4-Test models

    helper.test_model_RT(model, device,"PyTorch FP32")
    helper.test_model_RT(model.half(), device,"PyTorch FP16",in_half=True)

    helper.test_model_RT(model_rt, device,"TensorRT FP32")
    helper.test_model_RT(model_rt_half, device,"TensorRT FP16",in_half=True)
    helper.test_model_RT(model_rt_8bit, device,"TensorRT INT8",in_half=True)

