"""
This file defines the dataloader to load the images from GDEM DATASET to feed the model
Author:Alejandro
"""

import glob
import numpy as np
from PIL import Image
import os
import torch
import torch.utils.data as data
import cv2
from torchvision import transforms
#from dataloaders.occlusions_removal import slow_remove_occlusions

def read_rgb(path):
    rgb=cv2.imread(path)
    #print("RGB max value->"+str(rgb.max()))

    #print("SHAPE JUST READED RGGB----->"+str(rgb.shape))
    gray = np.array(Image.fromarray(rgb).convert('L'))
    gray = np.expand_dims(gray, -1)
    #print(rgb.shape)
    #print("RGB->"+str(rgb.shape))
    return rgb, gray


def read_depth(path, preprocess_depth=False):
    depth=cv2.imread(path)
    depth = np.array(Image.fromarray(depth).convert('L'))

    #if preprocess_depth:
    #    depth=slow_remove_occlusions(depth)
    depth=np.expand_dims(depth, 2)
    #print("DEpth max value->"+str(depth.max()))

    #depth=np.expand_dims(np.load(path)['arr_0'],2)
    #print(depth.shape)
    #print("DEPTH->"+str(depth.shape))
    return depth


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def ToTensor(img):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W

    Convert a ``numpy.ndarray`` to tensor.

    Args:
        img (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not (_is_numpy_image(img)):
        raise RuntimeError('AlEJANDRO ERROR-->img should be ndarray or dimesions are wrong. Got type (' +
                           str(type(img))+') and dimensions ('+str(img.shape)+')')

    if isinstance(img, np.ndarray):
        # handle numpy array
        if img.ndim == 3:
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
        elif img.ndim == 2:
            img = torch.from_numpy(img.copy())
        else:
            raise RuntimeError('AlEJANDRO ERROR-->img should be ndarray or dimesions are wrong. Got type (' +
                               str(type(img))+') and dimensions ('+str(img.shape)+')')
        return img


def get_paths(scene):
    """Returns the paths for the data files

    Args:
        train_or_test (["train"/"test"]): To load files from train or test directory

    Returns:
        [dict]: {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    """

    path_d = "data_gdem/scene_"+str(scene)+"/sparse/*.png"
    #path_gt = "data_gdem/scene_"+str(scene)+"/gt/*.png"
    path_rgb = "data_gdem/scene_"+str(scene)+"/rgb/*.png"

    paths_d = sorted(glob.glob(path_d))
    paths_rgb = sorted(glob.glob(path_rgb))
    #paths_gt = sorted(glob.glob(path_gt))

    if len(paths_d) == 0 or len(paths_rgb) == 0 :#or len(paths_gt) == 0:
        raise (RuntimeError(
            "ALEJANDRO ERROR--->No images in some or all of the paths of the dataloader.\n Len(RGB)="+str(len(paths_rgb))+"\n Len(Depth)="+str(len(paths_d))+"\n Len(gt)="+str(len(paths_gt))))
    #paths = {"rgb": paths_rgb, "d": paths_d, 'gt': paths_gt}
    paths = {"rgb": paths_rgb, "d": paths_d}
    return paths


class GdemDataLoader(data.Dataset):
    def __init__(self, scene=1):
        """Inits the Gdem dataloader

        Args:
            train_or_test (string): "train" or "test" to use different data (train and inference)
        """
        self.paths = get_paths(scene)

    def __getraw__(self, index):
        rgb, gray = read_rgb(self.paths['rgb'][index]) if (
            self.paths['rgb'][index] is not None) else None
        depth = read_depth(self.paths['d'][index], False) if (
            self.paths['d'][index] is not None) else None
        return rgb, gray, depth

    def __getitem__(self, index):
        rgb, gray, sparse = self.__getraw__(index)
        resize=transforms.Resize((1024,1024))
        
        preprocess_rgb=transforms.Compose([
            transforms.Resize((1024,1024)),
            #transforms.Normalize(mean=[0.4409, 0.4570, 0.3751], std=[0.2676, 0.2132, 0.2345])         
            transforms.Normalize(mean=[0,0,0], std=[1,1,1]) 
        ])
        preprocess_depth=transforms.Compose([
            transforms.Resize((1024,1024)),
            #transforms.Normalize(mean=[0.2674], std=[0.1949])
            transforms.Normalize(mean=[0], std=[1])          
        ])
        
        rgb=ToTensor(rgb).float()/255
        sparse=ToTensor(sparse).float()/255
        items={'rgb':preprocess_rgb(rgb), 'd':preprocess_depth(sparse)}
        
        #items = {"rgb": preprocess_rgb(ToTensor(rgb/255).float()), "d": preprocess_depth(ToTensor(sparse/255).float()), 'gt':preprocess_depth(ToTensor(gt/255).float()), 'g':resize(ToTensor(gray/255).float())}
        
        #def to_float_tensor(x): return resize(ToTensor(x).float())
        #candidates = {"rgb": rgb/255, "d": sparse/255, 'gt':gt/255, 'g':gray/255}
        #items = {
        #    key: to_float_tensor(val)
        #    for key, val in candidates.items() if val is not None
        #}
        return items

    def __len__(self):
        return len(self.paths['rgb'])
