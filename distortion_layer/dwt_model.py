import pywt
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from pytorch_wavelets import DWT1DForward as cudadwt
from pytorch_wavelets import DWT1DInverse as cudaidwt
from utils.hparameter import *

my_dwt = cudadwt(wave='db1').to(device)
my_idwt = cudaidwt(wave='db1').to(device)


def dwt_transform(aud_tensor):
    '''
    aud_tensor:[batch_size, 1(channel_size), 512*512(feature_dim)]
    ca_tensor:[batch_size, 1(channel_size), 512*512(feature_dim)]
    cb_list:[batch_size, 1(channel_size), 512*512(feature_dim)]
    '''
    
    '''
    my_dwt
    inputs:(N,C,L)
    ca:tensor(N,C,L/2)
    cb:list[tensor(N,C,L/2)]
    '''
    ca_tensor, cb_list = my_dwt(aud_tensor)
    ca_tensor = ca_tensor
    return ca_tensor, cb_list[0]


def dwt_inverse(ca_tensor, cb_tensor):
    '''
    ca_tensor:[batch_size, 1(channel_size), 512(feature_dim)]
    cb_list:[batch_size, 1(channel_size), 512(feature_dim)]
    aud_tensor:[batch_size, 1(channel_size), 512(feature_dim)]
    '''
    aud_tensor = my_idwt((ca_tensor, [cb_tensor]))
    aud_tensor = aud_tensor
    return aud_tensor


def cpu_dwt_transform(aud):
    ca, cb = pywt.dwt(aud, "db1")
    return ca, cb


def cpu_dwt_inverse(ca, cb):
    aud = pywt.idwt(ca, cb, "db1")
    return aud
