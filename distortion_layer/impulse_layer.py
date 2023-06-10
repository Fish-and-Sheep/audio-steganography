
import os
import torch
import random
import torchaudio
import numpy as np
from utils.hparameter import *
from torch_audiomentations import ApplyImpulseResponse,Compose

'''
Collect room impulse responses and convert them into 16kHz mono wav files
You can refer many databases from this script:
 [https://github.com/kaldi-asr/kaldi/blob/master/egs/aspire/s5/local/multi_condition/prepare_impulses_noises.sh]
 (https://github.com/kaldi-asr/kaldi/blob/master/egs/aspire/s5/local/multi_condition/prepare_impulses_noises.sh)
'''
class impulse_attack():
    def __init__(self, sr1, sr2):
        self.name = "impulse_attack_layer"
        irr_path = IRR_PATH
        files = os.listdir(irr_path)
        files.sort()
        count = 0
        for file in files:
            file = irr_path + '/' + file
            if count == 0:
                now_impulse = torch.mean(torchaudio.load(file)[
                                         0], dim=0).unsqueeze(0)
                self.impulse_all = [now_impulse]
            else:
                now_impulse = torch.mean(torchaudio.load(file)[
                                         0], dim=0).unsqueeze(0)
                self.impulse_all.append(now_impulse)
            count += 1
        self.nfft = 2 ** int(torch.ceil(torch.log(torch.tensor(NUMBER_SAMPLE)
                                             )/torch.log(torch.tensor(2))))


    def impulse(self, sig):
        '''
        sig:[batch,1,length]
        ae_convolved:[batch,1,length]
        '''
        #impulse_sig, sr1 = torchaudio.load('s1_r3_o.wav')
        #conv_length = impulse_sig.shape[1] + sig.shape[1] - 1
        #conv_length = sig.shape[2]
        #impulse_sig = torchaudio.functional.resample(impulse_sig, sr1, sr2)
        # nfft = 2 ** int(torch.ceil(torch.log(torch.tensor(conv_length)
        #                                      )/torch.log(torch.tensor(2))))
        i = random.choice([i for i in range(len(self.impulse_all))])
        print(f"selected {i}")
        selected_impulse = self.impulse_all[i].to(device)
        imp_filters = torch.fft.rfft(selected_impulse, n=self.nfft)
        # fft_length = np.array([nfft], dtype=np.int32)
        ae_frequency = torch.fft.rfft(sig, n=self.nfft)*imp_filters
        print(imp_filters.shape)
        ae_convolved = torch.fft.irfft(ae_frequency, n=self.nfft)[
            :, :, :NUMBER_SAMPLE]
        return ae_convolved


class impulse_attack2():
    def __init__(self, sr1, sr2):
        self.name = "impulse_attack_layer"
        self.irr_path = IRR_PATH #Collect room impulse responses and convert them into 16kHz mono wav files
        
    def impulse(self, sig):
        augment = Compose([
        ApplyImpulseResponse(self.irr_path,p=1,compensate_for_propagation_delay=True),
        ])
        ae_convolved = augment(sig, sample_rate=SAMPLE_RATE)
        return ae_convolved


if __name__ == "__main__":
    my = impulse_attack(1, 1)
    sig = torch.rand(4, 1, 512*512*2)
    ae = my.impulse(sig)
    print(ae.shape)
