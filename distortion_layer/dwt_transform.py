import torch
import numpy as np
from distortion_layer.dwt_model import *
from utils.hparameter import *


def aud_to_dwt(sample):
    '''
    sample:(audio_length)
    ca_tensor:[1, feature_length]
    '''
    sample = torch.FloatTensor(sample).unsqueeze(0)
    sample = torch.FloatTensor(sample).unsqueeze(0).to(device)

    ca_tensor, cb_tensor = dwt_transform(sample)
    ca_tensor = ca_tensor.squeeze(0)
    cb_tensor = cb_tensor.squeeze(0)
    # print(ca_tensor.shape)
    return ca_tensor, cb_tensor