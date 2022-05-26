import torch
from torch2trt import torch2trt
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

model=models.CNNModel_RT().to(device)
#model.eval()

example_rgb=torch.ones((1, 3, 1024, 1024)).to(device)
example_depth=torch.ones((1, 1, 1024, 1024)).to(device)

example_input=[example_rgb,example_depth]
example_input_2=torch.cat([example_rgb,example_depth],1)

#print(example_input_2.shape)
print("Converting model to TensorRT...")
# convert to TensorRT feeding sample data as input
model_rt=torch2trt(model,example_input, fp16_mode=True)
model.half()
print("Running TensorRT model...")
#Check agains pytorch
#y = model(example_rgb,example_depth)
#y_trt = model_rt(example_rgb,example_depth)
#
## check the output against PyTorch
#print(torch.max(torch.abs(y - y_trt)))

#Check inference times on both versions of the model
helper.test_model_RT(model_rt, device,"TensorRT",in_half=True)
helper.test_model_RT(model, device,"PyTorch",in_half=True)