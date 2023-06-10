
import torch
import os
import torchaudio
import numpy as np
import librosa
import cv2 as cv
import soundfile
import random
import pywt
import warnings
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image
from distortion_layer.dwt_transform import aud_to_dwt
from networks.encoder import redundant_encoder as RedundantEncoder
from networks.decoder import redundant_decoder as RedundantDecoder
from PIL import ImageFile, Image
from distortion_layer.attack_layer import attack_opeartion
from torch.nn.functional import mse_loss
from distortion_layer.dwt_model import dwt_inverse, dwt_transform
from utils.hparameter import CHANNEL_SIZE, NUMBER_SAMPLE, N_FFT, HOP_LENGTH, payload_size, device, data_depth, hidden_size, SAMPLE_RATE
from utils.test_parameters import MODEL_PATH, SELECT_MODEL_INDEX, TEST_ROOT
warnings.filterwarnings("ignore")
from utils.crc_bch import add_crc, new_add_bch
attack_layer = attack_opeartion(device)
import scipy.signal as signal
from my_xor_data import rand_256,rand_65536,rand_262144,wm,ustc,cumt
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


def the_stft(x, **params):
    f, t, zxx = signal.stft(x, nfft=N_FFT, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH) 
    return f, t, zxx

def stft_specgram(x, title, picname=None, **params):
    f, t, zxx = the_stft(x, **params)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if picname is not None:
        plt.savefig(str(picname) + '.jpg')
    plt.clf()
    return t, f, zxx

def residual_stft_specgram(x, y, title, picname=None, **params):
    f, t, zxx = the_stft(x, **params)
    f_y, t_y, zxx_y = the_stft(y, **params)
    zxx = zxx - zxx_y
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if picname is not None:
        plt.savefig(str(picname) + '.jpg')
    plt.clf()
    return t, f, zxx


def my_xor(a, b, length):
    result = int(a, 2) ^ int(b[:length], 2)
    result = bin(result)[2:].zfill(length)
    return result

def wm_to_tensor(wm_str):
    #data = bytearray(wm_str + '~'*(1245-len(wm_str)), 'utf-8')
    data = bytearray(wm_str, 'utf-8')
    data = ''.join(format(x, '08b') for x in data)
    #if len(data)<=256:data=my_xor(data,rand_256,len(data))
    #elif len(data)>256 and len(data)<=65536:data=my_xor(data,rand_65536,len(data))
    #elif len(data)>65536:
    data=my_xor(data,rand_262144,len(data)) 
    data = torch.Tensor([int(i) for i in data])
    # data = add_crc(data)
    data = new_add_bch(data)
    data = data.view(1,1,payload_size,payload_size)
    data = (data-0.5)*2
    return data


def tensor_to_img(img_tensor, img_path):
    img = np.array(img_tensor[0][0])
    # cv.imwrite(img_path, img.astype(np.uint8))
    return img


def do_encode(spectrum_path, encoder, cover, sr, watermark, out_path, dc, device, i, factor):
    with torch.no_grad():
        cover = cover[None].to(device) #[1, 1,feature_length]
        
        payload = wm_to_tensor(watermark).to(device)

        generated = encoder.forward(cover, payload, factor) #[1, 1,feature_length]
        dc = dc[None].to(device)
        aud_tensor = dwt_inverse(generated, dc) #[1, 1,feature_length]
        cover_aud_tensor = dwt_inverse(cover, dc) #[1, 1,feature_length]

        zero_tensor = torch.zeros(cover_aud_tensor.shape).to(device)
        cover_mse = mse_loss(cover_aud_tensor, zero_tensor, reduction='sum')
        encoder_mse = mse_loss(cover_aud_tensor, aud_tensor.clamp(-1,1), reduction='sum')
        snr = 10 * torch.log10(cover_mse / encoder_mse)

        # cover_audio_for_mel = cover_aud_tensor.cpu().squeeze(0).squeeze(0).detach().numpy()
        encoded_audio_for_mel = aud_tensor.cpu().squeeze(0).squeeze(0).detach().numpy()
        # stft_specgram(cover_audio_for_mel, 'Cover', picname=spectrum_path+'cover_audio_mel')
        # stft_specgram(encoded_audio_for_mel, 'Encoded', picname=spectrum_path+'encoded_audio_mel')
        # residual_stft_specgram(cover_audio_for_mel, encoded_audio_for_mel, 'Residual', picname=spectrum_path+"residual_audio_mel")
        
        soundfile.write(out_path, encoded_audio_for_mel, samplerate=sr)
        print(f"encode finished! encoder_mse:{encoder_mse} And snr:{snr}")
    return encoder_mse, snr, generated, payload, aud_tensor


def do_attack(generated_audio, sr, attack_choice, dc, device):
    attacked_data = attack_layer.attack(generated_audio, choice=attack_choice)
    attacked_audio = attacked_data.cpu().squeeze(0).squeeze(0).detach().numpy()
    soundfile.write('results/test_result/attacked_audio.wav', attacked_audio, samplerate=sr)
    carrier_reconst_tag, _ = dwt_transform(attacked_data)

    # with open('results/test_result/attacked_audio.wav', 'rb') as f:
    #     attacked2, sr = torchaudio.load(f.name)
    # attacked2 = attacked2[0][:NUMBER_SAMPLE][None][None].to(device)
    # carrier_reconst_tag2, _ = dwt_transform(attacked2)
    # import pdb
    # pdb.set_trace()
    return carrier_reconst_tag


def do_extract(decoder, attacked_data, payload):
    with torch.no_grad():
        decoded = decoder.forward(attacked_data)
        imag = decoded >= 0
        imag = imag.cpu()
        original_image = payload >= 0
        original_image = original_image.cpu()
        original_image = torch.tensor(original_image, dtype=torch.uint8)
        original_image = tensor_to_img(original_image, './original_image.jpg')
        decoder_loss = mse_loss(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(
            payload >= 0.0).sum().float() / payload.numel()  # .numel() calculate the number of element in a tensor
        print("Decoder loss: %.3f"% decoder_loss.item())
        print("Decoder acc: %.3f"% decoder_acc.item())
    return decoder_acc.item()


def test_main(attack_choice=1,
              audio_path='',
              watermark='.',
              audio_out_path='',
              spectrum_path = '',
              snr_file = None,
              factor=1,
              encoder=None,
              decoder=None):
    # load model file
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info("\nselected audio:{}\nembeded audio:{}".format(audio_path,audio_out_path))

    p = audio_path
    # load audio data and transform
    with open(p, 'rb') as f:
        sig, sr = torchaudio.load(f.name)
        if sr != SAMPLE_RATE:
            raise("sample rate should be 44.1k")
    sig = sig[0][:NUMBER_SAMPLE]
    ac, dc = aud_to_dwt(sig)

    # do encode
    out_path = audio_out_path
    encoder_mse, snr, generated, payload, generated_audio = do_encode(spectrum_path, encoder, ac, sr, watermark, out_path, dc, device, 1, factor)
    snr_file.write(f"audio_path={audio_path}\tsnr={snr}\n")
    # attacked_data = do_attack(generated_audio, sr, attack_choice, dc, device)
    # do extract
    # acc = do_extract(decoder, attacked_data, payload)
    return snr


def main(root = TEST_ROOT, out = './results/test_result', watermark=cumt, factor=1):
    SNR_FILE = os.path.join(out, 'snr.txt')
    AUDIO_OUT_PATH = os.path.join(out, "wm_audio/")
    SPECTRUM_PATH = os.path.join(out, "spectrum/")
    files = os.listdir(root)
    files.sort()
    audio_list = []
    for file_ in files:
        if file_[-4:] == ".wav":
            audio_list.append(file_)
    if not os.path.exists(AUDIO_OUT_PATH): os.makedirs(AUDIO_OUT_PATH)
    if not os.path.exists(SPECTRUM_PATH): os.makedirs(SPECTRUM_PATH)
    snr_file = open(SNR_FILE,'a+')
    model_path = MODEL_PATH
    model_list = []
    files = os.listdir(model_path)
    # files = sorted(files,key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
    for file_ in files:
        if file_[-4:] == ".dat":
            model_list.append(file_)
    PATH = os.path.join(model_path, model_list[SELECT_MODEL_INDEX])
    print(PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = RedundantEncoder(
        data_depth, hidden_size, CHANNEL_SIZE).to(device)
    decoder = RedundantDecoder(
        data_depth, hidden_size, CHANNEL_SIZE).to(device)
    # load net parameters
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location='cuda:0')
    else:
        checkpoint = torch.load(
            PATH, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint['state_dict_encoder'],strict=False)
    decoder.load_state_dict(checkpoint['state_dict_decoder'],strict=False)
    for attack_choice in [0]:
        avg_snr = 0
        for SELECT_AUDIO_NAME in audio_list:
            TEST_AUDIO = os.path.join(root,SELECT_AUDIO_NAME)
            EMBED_AUDIO = os.path.join(AUDIO_OUT_PATH, SELECT_AUDIO_NAME)
            spectrum_path = os.path.join(SPECTRUM_PATH, SELECT_AUDIO_NAME) + "_"
            snr = test_main(attack_choice=attack_choice,
                        audio_path=TEST_AUDIO,
                        watermark=watermark,
                        audio_out_path=EMBED_AUDIO,
                        spectrum_path = spectrum_path,
                        snr_file = snr_file,
                        factor = factor,
                        encoder=encoder,
                        decoder=decoder)
            avg_snr += snr
        avg_snr /= len(audio_list)
        snr_file.write(f"avg_snr={avg_snr}\n")
    snr_file.close()


if __name__ == "__main__":
    import fire
    fire.Fire(main)