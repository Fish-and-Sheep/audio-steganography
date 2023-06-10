import torch
import random
import torch.nn as nn
import torchaudio
import random
import julius
from utils.hparameter import *
from distortion_layer.impulse_layer import impulse_attack, impulse_attack2
from audiomentations import Compose, Mp3Compression
import kornia
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import logging
import soundfile
from pydub import AudioSegment
import os
import subprocess
import time
import multiprocessing as mp
import concurrent.futures

torchaudio.set_audio_backend("sox_io")

logging.basicConfig(level=logging.INFO, format='%(message)s')

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class attack_opeartion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.impulse_layer = impulse_attack2(sr1=10000, sr2=SAMPLE_RATE)
        self.bandpass = julius.BandPassFilter(1/44.1, 4/44.1).to(device)
        self.lowpass = julius.LowPassFilter(7.35/44.1).to(device)
        K = 0.9 
        self.resample1 = julius.ResampleFrac(SAMPLE_RATE, int(SAMPLE_RATE*K)).to(device)
        self.resample2 = julius.ResampleFrac(int(SAMPLE_RATE*K), SAMPLE_RATE).to(device)

        self.band_lowpass = julius.LowPassFilter(4/44.1).to(device)
        self.band_highpass = julius.HighPassFilter(1/44.1).to(device)

        self.drop_index = torch.ones(NUMBER_SAMPLE, device=self.device)
        i = 0
        while i < NUMBER_SAMPLE:
            self.drop_index[i] = 0.
            i += 100

        self.augment = Compose([Mp3Compression(p=1.0, min_bitrate=64, max_bitrate=64)])

    def white_noise(self, y): # SNR = 10log(ps/pn)
        # choice = [20, 50]
        choice = [20]
        SNR = random.choice(choice)
        mean = 0.
        RMS_s = torch.sqrt(torch.mean(y**2, dim=2))
        RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))
        for i in range(y.shape[0]):
            noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
            if i == 0:
                batch_noise = noise
            else:
                batch_noise = torch.cat((batch_noise, noise), dim=0)
        batch_noise = batch_noise.unsqueeze(1).to(self.device)
        signal_edit = y + batch_noise
        return signal_edit

    def band_pass(self, y):
        # y = self.lowpass(y)
        y = self.band_highpass(y)
        # y = self.band_lowpass(y)
        # y = self.bandpass(y)
        return y

    def resample(self, y):
        K = 0.9         
        y = self.resample1(y)
        y = self.resample2(y)
        y = y[:,:,:NUMBER_SAMPLE]
        return y

    def crop_out(self, y):
        return y*self.drop_index

    def change_top(self, y):
        y = y*0.9
        return y

    def recount(self, y):
        y2 = torch.tensor(np.array(y.cpu().squeeze(0).data.numpy()*(2**7)).astype(np.int8)) / (2**7)
        y2 = y2.to(self.device)
        y = y + (y2 - y)
        return y

    def medfilt(self, y):
        y = kornia.filters.median_blur(y.unsqueeze(1), (1, 3)).squeeze(1)
        return y
    
    def record(self, y):
        # https://github.com/adefossez/julius
        # Impulse Response
        if not IMPULSE_ABLATION:
            y = self.impulse_layer.impulse(y)
        if not BANDPASS_ABLATION:
            y = self.bandpass(y)
        if not NOISE_ABLATION:
            choice = [40, 50]
            # SNR = random.choice(choice)
            SNR = random.randint(40,50)
            mean = 0.
            RMS_s = torch.sqrt(torch.mean(y**2, dim=2)) 
            RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))
            for i in range(y.shape[0]):
                noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
                if i == 0:
                    batch_noise = noise
                else:
                    batch_noise = torch.cat((batch_noise, noise), dim=0)
            batch_noise = batch_noise.unsqueeze(1).to(self.device)
            y = y + batch_noise
        return y


    def record2(self, y, global_step):
        ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])
        if not IMPULSE_ABLATION:
            fre = torch.rand(1)[0] * ramp_fn(10000)
            y = self.impulse_layer.impulse(y,fre)

        if not NOISE_ABLATION:
            mean = 0.
            for i in range(y.shape[0]):
                RMS_n = torch.rand(1)[0] * ramp_fn(1000) * 0.02
                # noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
                noise = torch.normal(mean, float(RMS_n), size=(1, y.shape[2]))
                if i == 0:
                    batch_noise = noise
                else:
                    batch_noise = torch.cat((batch_noise, noise), dim=0)
            batch_noise = batch_noise.unsqueeze(1).to(self.device)
            y = y + batch_noise

        if not BANDPASS_ABLATION:
            # high_fre = torch.rand(1)[0] * ramp_fn(10000) * 8/44.1
            fre = torch.rand(1)[0] * ramp_fn(1000)
            if fre > 0.5:
                high_fre = (10 - torch.rand(1)[0] * ramp_fn(10000) * 2)
                band_lowpass = julius.LowPassFilter(high_fre/44.1).to(device)
                y = band_lowpass(y)
            else:
                pass
        return y

    
    def one_white_noise(self, y): # SNR = 10log(ps/pn)
        # choice = [20, 50]
        # SNR = random.choice(choice)
        SNR = random.randint(4,12)*5
        mean = 0.
        RMS_s = torch.sqrt(torch.mean(y**2, dim=2))  # RMS value of signal
        RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))  # RMS values of noise
        # Therefore mean=0, to round you can use RMS as STD
        for i in range(y.shape[0]):
            noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
            if i == 0:
                batch_noise = noise
            else:
                batch_noise = torch.cat((batch_noise, noise), dim=0)
        batch_noise = batch_noise.unsqueeze(1).to(self.device)
        signal_edit = y + batch_noise
        return signal_edit
    
    def two_band_pass(self, y):
        high = random.randint(4,8)
        self.bandpass = julius.LowPassFilter(high/44.1).to(device)
        y = self.bandpass(y)
        return y

    def record3(self, y):
        if not IMPULSE_ABLATION:
            y = self.impulse_layer.impulse(y)
        if not BANDPASS_ABLATION:
            y = self.two_band_pass(y)

        if not NOISE_ABLATION:
            y = self.one_white_noise(y)
        return y



    def attack_func(self, y, choice=None):
        '''
        y:[batch, 1, audio_length]
        out:[batch, 1, audio_length]
        '''
        if choice == None:
            return y
        elif choice == 1:
            return self.white_noise(y)
        elif choice == 2:
            return self.band_pass(y)
        elif choice == 3:
            return self.resample(y)
        elif choice == 4:
            return y
        elif choice == 5:
            return self.mp3(y)
        elif choice == 6:
            return self.crop_out(y)
        elif choice == 7:
            return self.change_top(y)
        elif choice == 8:
            return self.recount(y)
        elif choice == 9:
            return self.medfilt(y)
        elif choice == 10:
            return self.record(y)
        elif choice == 11:
             return self.compress_net(net,y)
        elif choice==13: # all attack
            ch = [1,3,4,5,6,7,8,9,10]
            ch2 = random.choice(ch)
            y = self.attack(y,choice=ch2)
            return y
        elif choice == 20:
            return self.record2(y)
        elif choice == 30:
            return self.record3(y)
        else:
            return y


    def attack(self, y, choice=None):
        y = y.clamp(-1,1)
        if choice==10:
            choice = np.random.choice([0,10])
        out = self.attack_func(y, choice=choice)
        return out.clamp(-1,1)
    

def mp3_compress(y,path):
    out=torch.zeros(y.shape)
    for i in range(y.shape[0]):
        sig=y[i, 0, :].cpu().detach()
        soundfile.write("{}/{}/{}.wav".format(path,torch.cuda.current_device(),i),sig,samplerate=44100)
        sig = AudioSegment.from_wav("{}/{}/{}.wav".format(path,torch.cuda.current_device(),i))
        sig.export("{}/{}/{}.mp3".format(path,torch.cuda.current_device(),i), format='mp3', bitrate='128k', parameters=['-ar', '44100'])
        sig_1=AudioSegment.from_mp3("{}/{}/{}.mp3".format(path,torch.cuda.current_device(),i))
        sig_1.export("{}/{}/{}.wav".format(path,torch.cuda.current_device()+4,i),format='wav',  parameters=['-ar', '44100'])
        with open("{}/{}/{}.wav".format(path,torch.cuda.current_device()+4,i), 'rb') as f:
            sig_2, sr = torchaudio.load(f.name)
            sig_2 = sig_2[0][:NUMBER_SAMPLE]
        out[i,0,:]=sig_2
    out=out.cuda()
    out=y+(out-y).detach()
    return out

        

def gaussian(y):
    std=1
    mean=0
    noise = torch.randn_like(y) * std + mean
    return noise+y



def flac_compress(y,path):
    out=torch.zeros(y.shape)
    for i in range(y.shape[0]):
        sig=y[i, 0, :].cpu().detach()
        soundfile.write("{}/{}/{}.wav".format(path,torch.cuda.current_device(),i),sig,samplerate=44100)
        sig = AudioSegment.from_wav("{}/{}/{}.wav".format(path,torch.cuda.current_device(),i))
        sig.export("{}/{}/{}.flac".format(path,torch.cuda.current_device(),i), format='flac', parameters=['-ar', '44100'])
        sig_1=AudioSegment.from_file("{}/{}/{}.flac".format(path,torch.cuda.current_device(),i))
        sig_1.export("{}/{}/{}.wav".format(path,torch.cuda.current_device()+4,i),format='wav',  parameters=['-ar', '44100'])
        with open("{}/{}/{}.wav".format(path,torch.cuda.current_device()+4,i), 'rb') as f:
            sig_2, sr = torchaudio.load(f.name)
            sig_2 = sig_2[0][:NUMBER_SAMPLE]
        out[i,0,:]=sig_2
    out=out.cuda()
    out=y+(out-y).detach()
    return out




def aac_compress(y,path):
    neroAacEnc_path="/public/chenkj/audio/neroAacEnc"
    neroAacDec_path="/public/chenkj/audio/neroAacDec"
    out=torch.zeros(y.shape)
    for i in range(y.shape[0]):
        sig=y[i, 0, :].cpu().detach()
        soundfile.write("{}/{}/{}.wav".format(path,torch.cuda.current_device(),i),sig,samplerate=44100)
        input_file="{}/{}/{}.wav".format(path,torch.cuda.current_device(),i)
        output_file="{}/{}/{}.aac".format(path,torch.cuda.current_device(),i)
        subprocess.run('nohup "{}" -if "{}" -of "{}" -br 128000 -ignorelength -ifileformat wav -ofmt raw -ss 44100 > /dev/null 2>&1 &'.format(neroAacEnc_path, input_file, output_file), shell=True, check=False)
        time.sleep(5)
        #os.system('nohup "${neroAacEnc_path}" -if "${input_file}" -of "${output_file}" -br 320000 -ignorelength -ifileformat wav -ofmt raw -ss 44100 > /dev/null 2>&1 &')
        input_file="{}/{}/{}.aac".format(path,torch.cuda.current_device(),i)
        output_file="{}/{}/{}.wav".format(path,torch.cuda.current_device()+4,i)
        os.system('nohup "{}" -if "{}" -of "{}" > /dev/null 2>&1 &'.format(neroAacDec_path, input_file, output_file))
        time.sleep(5)
        with open("{}/{}/{}.wav".format(path,torch.cuda.current_device()+4,i), 'rb') as f:
            sig_2, sr = torchaudio.load(f.name)
            sig_2 = sig_2[0][:NUMBER_SAMPLE]
        out[i,0,:]=sig_2
    out=out.cuda()
    out=y+(out-y).detach()
    return out


def aac_compress_sp(y,path):
    neroAacEnc_path="/public/chenkj/audio/neroAacEnc"
    neroAacDec_path="/public/chenkj/audio/neroAacDec"
    out=torch.zeros(y.shape)
    for i in range(y.shape[0]):
        sig=y[i, 0, :].cpu().detach()
        soundfile.write("{}/{}/{}.wav".format(path,torch.cuda.current_device(),i),sig,samplerate=44100)
        input_file="{}/{}/{}.wav".format(path,torch.cuda.current_device(),i)
        output_file="{}/{}/{}.aac".format(path,torch.cuda.current_device(),i)
        subprocess.run('nohup "{}" -if "{}" -of "{}" -br 128000 -ignorelength -ifileformat wav -ofmt raw -ss 44100 > /dev/null 2>&1 &'.format(neroAacEnc_path, input_file, output_file), shell=True, check=False)
    time.sleep(5)
    for i in range(y.shape[0]):
        input_file="{}/{}/{}.aac".format(path,torch.cuda.current_device(),i)
        output_file="{}/{}/{}.wav".format(path,torch.cuda.current_device()+4,i)
        os.system('nohup "{}" -if "{}" -of "{}" > /dev/null 2>&1 &'.format(neroAacDec_path, input_file, output_file))
    time.sleep(5)
    for i in range(y.shape[0]):
        with open("{}/{}/{}.wav".format(path,torch.cuda.current_device()+4,i), 'rb') as f:
            sig_2, sr = torchaudio.load(f.name)
            sig_2 = sig_2[0][:NUMBER_SAMPLE]
        out[i,0,:]=sig_2
    out=out.cuda()
    out=y+(out-y).detach()
    return out




def compress_attack(y,training,step,net,path):
    y=y.clamp(-1,1)
    if training:
        #if step%2==0:
        #out=mp3_compress(y,path)
        #out=flac_compress(y,path)
            #logging.info("训练真实压缩")
        #else:
        #band_lowpass = julius.LowPassFilter(20/44.1).to(device)
        #band_highpass=julius.HighPassFilter(0.02/44.1).to(device)

        #out=net(y)
        #out=mp3_compress(y,path)
        #out=aac_compress(y,path)

        out=y

        #mid_1=gaussian(mid)
        #mid_2=band_lowpass(mid_1)
        #out=band_highpass(mid_2)
            #logging.info("训练模拟压缩")
    else:
        #out=mp3_compress(y,path)
        #out=flac_compress(y,path)
        #out=mp3_compress(y,path)

        #out=aac_compress(y,path)

        out=y
        #logging.info("验证真实压缩")
    return out.clamp(-1,1)




