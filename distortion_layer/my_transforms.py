import torch
import numpy as np
from pytorch_wavelets import DWT1DForward as dwtmodel
#from distortion_layer.dwt_model import *
from utils.hparameter import *
cpudwt = dwtmodel(wave="db1")

def cpudwttransform(sample):
    ca,cblist = cpudwt(sample)
    return ca,cblist[0]


class my_transform(object):
    def __init__(self):
        self.name = "my_transform"

    def __call__(self, sample):
        '''
        sample:(audio_length)
        ca_tensor:[1, feature_length]
        '''
        sample = torch.FloatTensor(sample).unsqueeze(0) # 扩充channel
        sample = torch.FloatTensor(sample).unsqueeze(0)# 扩充batch

        ca_tensor, cb_tensor = cpudwttransform(sample)
        ca_tensor = ca_tensor.squeeze(0)
        cb_tensor = cb_tensor.squeeze(0)
        # print(ca_tensor.shape)
        return ca_tensor, cb_tensor