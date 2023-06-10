import torch

data_dir = "/public/jiangyichuan/Unet"
epochs = 1000
BATCH = 8
SAVE_MODEL_PATH = '最终/没有鉴别器_65536'


TRAIN_DATA = 10000
VAL_DATA = 500
WEIGHT_E = 10.*15.
WEIGHT_D = 1.*1.
WEIGHT_A = 0.01
ATTACK_TRAIN = [0]
CHANNEL_SIZE = 1


payload_length = 65536
pre_model_name = "_pre.dat"
payload_size = 256
print_each = 50
pre_discrim = 0 
load_pre = False
save_circle = 10
pre_model_path_ = SAVE_MODEL_PATH + '/last_model.dat'
data_depth = 2
hidden_size = 32
pre_epochs = 0


IRR_PATH = './utils/irr/'
CRC_LENGTH = 16
CRC_MODULE = 'crc-16'
BCH_POLYNOMIAL = 137 #285
BCH_BITS = 5
N_FFT = 1022
HOP_LENGTH = 502
AUDIO_LEN = 16
SAMPLE_RATE = 44100
NUMBER_SAMPLE= 524288
LOAD_MODEL = False
PATH=''
IMPULSE_ABLATION = False
BANDPASS_ABLATION = False
NOISE_ABLATION = False

Instance_Norm = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_workers = 0
cuda_count=torch.cuda.device_count()
