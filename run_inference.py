#Load dataloader->
#Load model->
#Measure inference time->
#Measure resources consumption->
"""
Snippet to correctly measure inference time on DNNs  (https://deci.ai/blog/measure-inference-time-deep-neural-networks/)  ->
model = EfficientNet.from_pretrained('efficientnet-b0')
device = torch.device("cuda")
model.to(device)
dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)

"""
"""
Other metrics can be obtained with THOP:Pytorch-OpCounter (https://github.com/Lyken17/pytorch-OpCounter)
Or using torchstat (not updated since 2018)
Measure memory footprint with Pytorch-Memory-Utils (https://github.com/Oldpan/Pytorch-Memory-Utils)
"""
"""
Follow the tutorial in order to run models in TensorRT and speed-up inference->
https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/?ck_subscriber_id=883632583
"""
import torch
import helpers.models as models
import helpers.helper as helper
 
cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))


#1- Load model
model_name="CNNModel_LateFusion"
resnet=models.ResNetModel().to(device)
inception=models.InceptionModel().to(device)
inception_late=models.InceptionModel_LateFusion().to(device)
cnn=models.CNNModel().to(device)
cnn_late=models.CNNModel_LateFusion().to(device)
attention=models.AttentionModel().to(device)

#2- Load Weights
#model.half()
#model.load_state_dict(torch.load("weights/CNNModel_LateFusion.pt", map_location=device))

#model.cuda().half()

#3- Begin tests
model_name_list=["ResNetModel","InceptionModel","InceptionModel_LateFusion","CNNModel","CNNModel_LateFusion","AttentionModel"]
model_list=[resnet,inception, inception_late,cnn,cnn_late,attention]

for model_name,model in zip(model_name_list,model_list):
    model.load_state_dict(torch.load("weights/"+str(model_name)+".pt", map_location=device))
    helper.test_model(model, device,model_name,in_half=False)
    model.half()
    helper.test_model(model, device,model_name,in_half=True)