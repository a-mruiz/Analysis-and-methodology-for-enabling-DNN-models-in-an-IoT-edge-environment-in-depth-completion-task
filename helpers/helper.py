import time
import deepCABAC
import numpy as np
from tqdm import tqdm
import torch
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from helpers.dataloaderMiddlebury import MiddleburyDataLoader
from helpers.dataloaderGDEM import GdemDataLoader
import helpers.losses as losses
cmap = plt.cm.binary

def save_image_color(img_merge, filename):
    cv2.imwrite(filename, img_merge)

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    
    return depth.astype('uint8')   

def decode_model_weights(model,weigths_path='weights/compressed_weights.bin'):
    """Receives a PyTorch model and outputs the same model with weights assigned and the time taken to do it

    Args:
        model (PyTorchModel): model
        weights_path (str): path to the model

    Returns:
        PyTorchModel: model with loaded weights
        float: elapsed time to decompress and load weights (s)
    """
    print("\nLoading decoder...")
    ini_time=time.time()
    decoder = deepCABAC.Decoder()
    with open(weigths_path, 'rb') as f:
        stream = f.read()
    decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
    state_dict = model.state_dict()
    
    print("Decoding and assigning weights...")
    for name in tqdm(state_dict.keys()):
        if '.num_batches_tracked' in name:
            continue
        param = decoder.decodeWeights()
        state_dict[name] = torch.tensor(param)
    decoder.finish()

    model.load_state_dict(state_dict)
    end_time=time.time()-ini_time
    print("OK")
    print("Time taken to decode and load weights (s)->"+str(end_time))
    return model,end_time    
    

def save_result_row(batch_data, output, name, folder="outputs/",azure_run=None):
    """Will save a row with the different images rgb+depth+gt+output

    Args:
        batch_data ([type]): [description]
        output ([type]): [description]
    """

    #unorm_rgb = transforms.Normalize(mean=[-0.4409/0.2676, -0.4570/0.2132, -0.3751/0.2345],
    #                         std=[1/0.2676, 1/0.2132, 1/0.2345])
    #unorm_d = transforms.Normalize(mean=[-0.2674/0.1949],
    #                         std=[1/0.1949])
    #unorm_gt = transforms.Normalize(mean=[-0.3073/0.1761],
    #                         std=[1/0.1761])

    unorm_rgb=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
    unorm_d=transforms.Normalize(mean=[0],std=[1])
    unorm_gt=transforms.Normalize(mean=[0],std=[1])

    rgb=unorm_rgb(batch_data['rgb'][0, ...])
    depth=unorm_d(batch_data['d'].squeeze_(0))
    gt=unorm_gt(batch_data['gt'].squeeze_(0))
    output=unorm_d(output.squeeze_(0))
    #depth=unorm_d(batch_data['d'])

    #rgb=batch_data['rgb'][0,...]
    #depth=batch_data['d']
    #gt=batch_data['gt']



    #print("OUTPUT Size------------------->"+str(output.shape))
    img_list=[]
    
    rgb = np.squeeze(rgb.data.cpu().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))*255
    img_list.append(rgb)

    depth = depth_colorize(np.squeeze(depth.data.cpu().numpy()))
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)

    gt = depth_colorize(np.squeeze(gt.data.cpu().numpy()))
    img_list.append(gt)

    #print("OUTPUT SIZE BEFORE--->"+str(output.shape))
    output = depth_colorize(np.squeeze(output.data.cpu().numpy()))
    #output = np.moveaxis(np.squeeze(output.data.cpu().numpy()),0,2)
    #print("OUTPUT SIZE--->"+str(output.shape))
    img_list.append(output)
    
    img_merge_up = np.hstack([img_list[0], img_list[2]])
    img_merge_down = np.hstack([img_list[1], img_list[3]])
    img_merge = np.vstack([img_merge_up, img_merge_down])
    img_merge= img_merge.astype('uint8')
    if azure_run:
        imgplot = plt.figure()
        plt.imshow(img_merge)
        
        azure_run.log_image(name=name,plot=imgplot)
    else:
        save_image_color(img_merge,folder+name)
    #print("saving img to "+str(folder+name))

   
def test_model(model,device,tag="",scene=1,in_half=False):
    """Will measure model times in inference

    Args:
        model (_type_): _description_
        device (_type_): _description_
        tag (str, optional): _description_. Defaults to "".
        scene (int, optional): _description_. Defaults to 1.
    """
    #print("\nLoading data to eval...")
    model.eval()
    #loading data to test
    dataset_test = GdemDataLoader(scene)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=False,
        sampler=None)
    #inference
    print("\n-Testing inference times over "+str(tag)+" ...")
    for i, batch_data in enumerate(test_dataloader):
        if in_half:
            batch_data = {
                key: val.to(device).half() for key, val in batch_data.items() if val is not None
            }
        else:
            batch_data = {
                key: val.to(device) for key, val in batch_data.items() if val is not None
            }
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 100
        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        print("\tWarming up GPU...")
        for _ in range(15):
            _ = model(batch_data)
        # MEASURE PERFORMANCE
        print("\tMeasuring times over "+str(repetitions)+" iterations...")
        with torch.no_grad():
            for rep in tqdm(range(repetitions)):
                starter.record()
                _ = model(batch_data)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                if rep>9:
                    timings[rep] = curr_time
        mean_syn = np.sum(timings) / (repetitions-10)
        std_syn = np.std(timings)
        if in_half:
            print(f"[MODEL IN FP16] Inference time mean ({mean_syn}) and std ({std_syn})  in ms")
        else:
            print(f"[MODEL IN FP32] Inference time mean ({mean_syn}) and std ({std_syn})  in ms")

        break
        #print(mean_syn)
def test_model_RT(model,device,tag="",scene=1,in_half=False):
    """Will measure model times in inference

    Args:
        model (_type_): _description_
        device (_type_): _description_
        tag (str, optional): _description_. Defaults to "".
        scene (int, optional): _description_. Defaults to 1.
    """
    #print("\nLoading data to eval...")
    model.eval()
    #inference
    print("\nTesting inference times over "+str(tag)+" ...")

    if in_half:
        example_rgb=torch.ones((1, 3, 1024, 1024)).to(device).half()
        example_depth=torch.ones((1, 1, 1024, 1024)).to(device).half()

    else:
        example_rgb=torch.ones((1, 3, 1024, 1024)).to(device)
        example_depth=torch.ones((1, 1, 1024, 1024)).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    print("\t-Warming up GPU...")
    for _ in range(15):
        _ = model(example_rgb,example_depth)
    # MEASURE PERFORMANCE
    print("\t-Measuring times over "+str(repetitions)+" iterations...")
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(example_rgb,example_depth)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            if rep>9:
                timings[rep] = curr_time
    mean_syn = np.sum(timings) / (repetitions-10)
    std_syn = np.std(timings)
    if in_half:
        print(f"\t-[MODEL IN FP16] Inference time mean ({mean_syn}) and std ({std_syn})  in ms")
    else:
        print(f"\t-[MODEL IN FP32] Inference time mean ({mean_syn}) and std ({std_syn})  in ms")
    
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = model(example_rgb,example_depth)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()
    fps = 50.0 / (t1 - t0)
    print("\t-Model Throughput (FPS)->"+str(fps))


    """
    NOW TEST ALL THE MODELS WITH THE MIDDLEBURY DATA IN ORDER TO OBTAIN QUALITY METRICS
    """
    dataset_test = MiddleburyDataLoader()
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=None)
    #val_losses=[]
    
    with torch.no_grad():
        val_psnrs=[]
        val_mses=[]
        mse_loss=losses.MaskedMSELoss()
        criterion=losses.CombinedNew()
        for i,batch_data in enumerate(test_dataloader):
            if in_half and "TensorRT" not in tag:
                batch_data=[val.to(device).half() for val in batch_data if val is not None]
            else:
                batch_data=[val.to(device) for val in batch_data if val is not None]
            output=model(*batch_data)
            #val_current_loss = criterion(output, batch_data[2]).item()
            #print("Ouput shape->"+str(output.shape))
            #print("GT shape->"+str(batch_data[2].shape))
            #val_current_mse = mse_loss(output,batch_data[2]).item()
            val_current_psnr = losses.psnr_loss(output, batch_data[2]).item()
            
            #val_current_mse = mse_loss(torch.squeeze(output,0), torch.squeeze(batch_data[2],0)).item()
            #val_losses.append(val_current_loss)
            val_psnrs.append(val_current_psnr)
            #val_mses.append(val_current_mse)
    #val_mean_loss= sum(val_losses)/len(val_losses)
    val_mean_psnr= -sum(val_psnrs)/len(val_psnrs)
    #val_mean_mse= sum(val_mses)/len(val_mses)
    #print(f"\t-Quality PSNR ({val_mean_psnr}), mse ({val_mean_mse})")
    print(f"\t-Quality PSNR ({val_mean_psnr})")
        
def compress_model_weights(path, params,save_route):
    model = torch.load(path,map_location=torch.device("cpu"))
    encoder = deepCABAC.Encoder()
    enc_time=time.time()
    
    for name, param in tqdm(model.items()):
        if '.num_batches_tracked' in name:
            continue
        param = param.cpu().numpy()
        if '.weight' in name:
            encoder.encodeWeightsRD(param, params['interv'], params['stepsize'], params['_lambda'])
        else:
            encoder.encodeWeightsRD(param, params['interv'], params['stepsize_other'], params['_lambda'])

    stream = encoder.finish().tobytes()

    uncompressed_size=1e-6 * os.path.getsize(path)
    compressed_size=1e-6 * len(stream)
    print("Uncompressed size: {:2f} MB".format(uncompressed_size))
    print("Compressed size: {:2f} MB".format(compressed_size))
    print("Compression ratio: "+str(uncompressed_size/compressed_size))
    enc_time=time.time()-enc_time
    print("Encoding time (s)=> "+str(enc_time))

    stream = encoder.finish()

    print("Saving encoded model to (weights.bin)")
    with open(save_route, 'wb') as f:
        f.write(stream)
