
import torch
import os
import torchaudio
import numpy as np
import cv2 as cv
import warnings
from distortion_layer.dwt_transform import aud_to_dwt
from networks.encoder import redundant_encoder as RedundantEncoder
from networks.decoder import redundant_decoder as RedundantDecoder
from distortion_layer.attack_layer import attack_opeartion
from torch.nn.functional import mse_loss
from distortion_layer.dwt_model import dwt_inverse, dwt_transform
warnings.filterwarnings("ignore")
from utils.crc_bch import add_crc, new_add_bch, verify_crc, do_ec
from utils.hparameter import CHANNEL_SIZE, NUMBER_SAMPLE, N_FFT, HOP_LENGTH, payload_size, device, data_depth, hidden_size, SAMPLE_RATE
from utils.test_parameters import MODEL_PATH, SELECT_MODEL_INDEX, TEST_ROOT
import logging
import julius
from my_xor_data import rand_256,rand_65536,rand_262144,wm,ustc,cumt

logging.basicConfig(level=logging.INFO, format='%(message)s')
attack_layer = attack_opeartion(device)

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
    cv.imwrite(img_path, img.astype(np.uint8))
    return img


def search_do_extract(attacked_path, watermak, decoder, len_file):
    p = attacked_path
    with open(p, 'rb') as f:
        raw_sig, sr = torchaudio.load(f.name)
        if sr != SAMPLE_RATE:
            raise("sample rate should be 44.1k")
            print("resample")
            resampler = julius.ResampleFrac(sr, SAMPLE_RATE)
            raw_sig = resampler(raw_sig)
    max_start = raw_sig.shape[1] - NUMBER_SAMPLE
    max_len = int(-3000) # microphone
    start_len = int(-8000) # microphone
    # max_len = int(-8000) # womic cell phone
    # start_len = int(-14000) # womic cell phone
    payload = wm_to_tensor(watermak).to(device)
    best_len_list = []
    
    for i_len in range(start_len, max_len):
        if i_len > 0:
            sig = torch.cat([torch.zeros(1,i_len),raw_sig], dim=1)
            sig = sig[0][:NUMBER_SAMPLE]
        else:
            if max_start + i_len < 0:
                sig = torch.cat([raw_sig,torch.zeros(1,-i_len-max_start)], dim=1)
                sig = sig[0][-i_len:NUMBER_SAMPLE-i_len]
            else:
                sig = raw_sig[0][-i_len:NUMBER_SAMPLE-i_len]
        spect, phase = aud_to_dwt(sig)
        with torch.no_grad():
            decoded = decoder.forward(spect[None])
            # decoder_loss = mse_loss(decoded, payload)
            decoder_acc = (decoded >= 0.0).eq(
                            payload >= 0.0).sum().float() / payload.numel()  # .numel() calculate the number of element in a tensor
            if round(decoder_acc.item(),2)==1:
                best_len_list.append(decoder_acc.item())
                break
            
          
            best_len_list.append(decoder_acc.item())
    best_len = best_len_list.index(max(best_len_list)) + start_len
    

    if best_len > 0:
        sig = torch.cat([torch.zeros(1,best_len),raw_sig], dim=1)
        sig = sig[0][:NUMBER_SAMPLE]
    else:
        if max_start + best_len < 0:
            sig = torch.cat([raw_sig,torch.zeros(1,-best_len-max_start)], dim=1)
            sig = sig[0][-best_len:NUMBER_SAMPLE-best_len]
        else:
            sig = raw_sig[0][-best_len:NUMBER_SAMPLE-best_len]

    spect, phase = aud_to_dwt(sig)
    with torch.no_grad():
        decoded = decoder.forward(spect[None])
        decoder_loss = mse_loss(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(
            payload >= 0.0).sum().float() / payload.numel()  # .numel() calculate the number of element in a tensor
        logging.info("shift:{}".format(best_len))
        logging.info("Decoder loss: %.3f"% decoder_loss.item())
        logging.info("Decoder acc: %.3f"% decoder_acc.item())
        # decoded_ = decoded.cpu()[0][0].view(1, payload_size*payload_size).detach().numpy()
        # payload_ = payload.cpu()[0][0].view(1, payload_size*payload_size).detach().numpy()
        # corr = np.corrcoef(decoded_, payload_)
        # logging.info("corr:\n{}".format(corr))
        decoded = (decoded >= 0.0).float()
        decoded_ec = do_ec(decoded.view(payload_size*payload_size))
        # verify = verify_crc(decoded_ec)
        len_file.write("{}\t{}\{}\n".format(attacked_path,decoder_acc,decoder_loss,0-best_len))
    return decoder_acc


def extract(attacked_path, watermak, decoder, len_file):
    p = attacked_path
    torchaudio.set_audio_backend("sox_io")
    # load audio data and transform
    with open(p, 'rb') as f:
        raw_sig, sr = torchaudio.load(f.name)
        if sr != SAMPLE_RATE:
            print("resample")
            resampler = julius.ResampleFrac(sr, SAMPLE_RATE)
            raw_sig = resampler(raw_sig)
    payload = wm_to_tensor(watermak).to(device)
    
    sig = raw_sig
    sig = sig[0][:NUMBER_SAMPLE]
    spect, phase = aud_to_dwt(sig)
    with torch.no_grad():
        decoded = decoder.forward(spect[None])
        # decoder_loss = mse_loss(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(
                        payload >= 0.0).sum().float() / payload.numel()  # .numel() calculate the number of element in a tensor
        decoded = (decoded >= 0.0).float()
        decoded_ec = do_ec(decoded.view(payload_size*payload_size))
        verify = verify_crc(decoded_ec)
    
        decoder_loss = mse_loss(decoded, payload)
        logging.info("Decoder loss: %.3f"% decoder_loss.item())
        logging.info("Decoder acc: %.3f"% decoder_acc.item())
        # decoded_ = decoded.cpu()[0][0].view(1, payload_size*payload_size).detach().numpy()
        # payload_ = payload.cpu()[0][0].view(1, payload_size*payload_size).detach().numpy()
        # corr = np.corrcoef(decoded_, payload_)
        # logging.info("corr:\n{}".format(corr))
        len_file.write("{}\t{}\{}\t{}\n".format(attacked_path,decoder_acc,decoder_loss,0))
    return decoder_acc


def main(attack=0, root="./results/test_result/attacked/", acc_dir="./results/test_result/", watermark = cumt):
    """_summary_

    Args:
        distance (str): the disatance of re_record scene
    """
    #---------------------------get model--------------------------
    model_path = MODEL_PATH
    model_list = []
    files = os.listdir(model_path)
    files = sorted(files,key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
    for file_ in files:
        if file_[-4:] == ".dat":
            model_list.append(file_)

    PATH = os.path.join(model_path, model_list[SELECT_MODEL_INDEX])
    logging.info("model:\n{}".format(PATH))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define net structure
    decoder = RedundantDecoder(
        data_depth, hidden_size, CHANNEL_SIZE).to(device)
    # load net parameters
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location='cuda:0')
    else:
        checkpoint = torch.load(
            PATH, map_location=lambda storage, loc: storage)
    decoder.load_state_dict(checkpoint['state_dict_decoder'],strict=False)

    #---------------------------get AUDIO--------------------------
    files = os.listdir(root)
    files.sort()
    audio_list = []
    for file_ in files:
        if file_[-4:] == ".wav":
            audio_list.append(file_)
    ACC_FILE = os.path.join(acc_dir, "acc.txt")
    acc_file = open(ACC_FILE,'a+')
    #---------------------------get WATERMARK--------------------------
    #---------------------------Extract watermark--------------------------
    avg_acc = 0
    for SELECT_AUDIO_NAME in audio_list:
        attacked_path = os.path.join(root, SELECT_AUDIO_NAME)
        logging.info("\nattacked audio:{}".format(attacked_path))
        if attack:
            acc = search_do_extract(attacked_path, watermark, decoder, acc_file)
        else:
            acc = extract(attacked_path, watermark, decoder, acc_file)
        avg_acc += acc
    avg_acc /= len(audio_list)
    acc_file.write(f"avg_acc:{avg_acc}\n")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
    
    
    
