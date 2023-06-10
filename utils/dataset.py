import os
import librosa
import torchaudio
from utils.hparameter import *
from torchvision import datasets, transforms
from distortion_layer.my_transforms import my_transform
from torch.utils.data import Dataset

def wav_loader(path):
    with open(path, 'rb') as f:
        sig, sr = torchaudio.load(f.name)
        sig = sig[0][:NUMBER_SAMPLE]
        return sig, sr


class my_dataset(Dataset):
    def __init__(self, root):
        self.dataset_path = os.path.expanduser(root)
        self.wavs = self.process_meta()
        if "train" in root:
            self.wavs = self.wavs[:TRAIN_DATA]
        elif "val" in root:
            self.wavs = self.wavs[:VAL_DATA]
        self.transform = my_transform()

        self.data = []
        for idx in range(len(self.wavs)):
            path = self.wavs[idx]
            print(f"path is : {path}")
            sample, sr = wav_loader(path)  # custom
            if self.transform is not None:
                sample = self.transform(sample)
                ca = sample[0]
                cb = sample[1]
            self.data.append((ca, cb))

    def __getitem__(self, index):
        a = self.data[index]
        return a[0], a[1]

    def __len__(self):
        return len(self.wavs)

    def process_meta(self):
        wavs = []
        wavs_name = os.listdir(self.dataset_path)
        for name in wavs_name:
            wavs.append(os.path.join(self.dataset_path,name))
        return wavs


# class my_dataset(Dataset):
#     def __init__(self, root):
#         self.dataset_path = os.path.expanduser(root)
#         self.wavs = self.process_meta()
#         self.transform = my_transform

#     def __getitem__(self, index):
#         path = self.wavs[index]
#         sample, sr = wav_loader(path)  # custom
#         if self.transform is not None:
#             sample = self.transform(sample)
#             ca = sample[0]
#             cb = sample[1]
#         return ca, cb

#     def process_meta(self):
#         wavs = []
#         wavs_name = os.listdir(self.dataset_path)
#         for name in wavs_name:
#             wavs.append(os.path.join(self.dataset_path,name))
#         return wavs
