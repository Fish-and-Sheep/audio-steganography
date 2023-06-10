# -*- coding: utf-8 -*-
import os
import gc
import sys
import torch
import os.path
import random
import datetime
import argparse
import logging
import numpy as np
import torch.nn as nn
from utils.hparameter import *
from distortion_layer.dwt_model import dwt_inverse, dwt_transform
from torch.optim import Adam
from rich.progress import track
from distortion_layer.attack_layer_self_net import compress_attack
from torchvision import transforms
from networks.discriminator import base_discriminator as Discriminator
from networks.encoder import redundant_encoder as Encoder
from networks.decoder import redundant_decoder as Decoder
from utils.dataset import my_dataset
from distortion_layer.my_transforms import my_transform
from distortion_layer.dwt_transform import aud_to_dwt
from torch.nn.functional import mse_loss
import warnings
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from distortion_layer.unet import unet
import datetime
import random

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.info("device: {}".format(device))
logging.info("emebdding rate: {}".format(payload_length))
device_ids = eval("[" + os.environ['CUDA_VISIBLE_DEVICES']+"]")



def save_model(encoder, decoder, en_de_optimizer, de_optimizer, discriminator, cr_optimizer, metrics, ep, attack_number, avg_acc, count_epoch):
    now = datetime.datetime.now()
    name = "epoch_%s_%s_%s_%s.dat" % (count_epoch, attack_number,now.strftime("%Y-%m-%d_%H_%M_%S"), avg_acc[:6])
    fname = SAVE_MODEL_PATH + '/' + name
    states = {
        'state_dict_critic': discriminator.module.state_dict(),
        'state_dict_encoder': encoder.module.state_dict(),
        'state_dict_decoder': decoder.module.state_dict(),
        'en_de_optimizer': en_de_optimizer.state_dict(),
        'cr_optimizer': cr_optimizer.state_dict(),
        'de_optimizer': de_optimizer.state_dict(),
        'metrics': metrics,
        'train_epoch': ep,
        'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
    }
    torch.save(states, fname)
    torch.save(states, pre_model_path_)


def save_pre_model(encoder, decoder):
    now = datetime.datetime.now()
    name = pre_model_name
    fname = SAVE_MODEL_PATH+name
    states = {
        'state_dict_encoder': encoder.state_dict(),
        'state_dict_decoder': decoder.state_dict()
    }
    torch.save(states, fname)


def train_model(scheduler,encoder, decoder, en_de_optimizer, de_optimizer, discriminator, cr_optimizer, metrics, train_loader, valid_loader):
    encoder.train()
    decoder.train()
    discriminator.train()
    bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
    encoded_label = 1.
    cover_label = 0.
    if TRAIN_DATA % BATCH == 0:
        train_steps_in_epoch = TRAIN_DATA // BATCH
    else:
        train_steps_in_epoch = TRAIN_DATA // BATCH + 1
    train_steps_in_epoch = TRAIN_DATA // BATCH

    for i in ATTACK_TRAIN:
        logging.info("*"*40 + "attack choice is: {}".format(i))
        encoder.to(device)
        decoder.to(device)
        discriminator.to(device)
        if load_pre:
            path ="/public/jiangyichuan/DeAR/DeAR_source_code/A_LAST_TEST/密文长度_256_不抗攻击_网络_异或_100%/epoch_43_0_2023-04-09_20_34_30_0.9999.dat"
            checkpoint = torch.load(path)
            encoder_state_dict=checkpoint['state_dict_encoder']
            decoder_state_dict=checkpoint['state_dict_decoder']
            discriminator_state_dict = checkpoint['state_dict_critic']
            encoder.load_state_dict(encoder_state_dict)
            decoder.load_state_dict(decoder_state_dict)
            discriminator.load_state_dict(discriminator_state_dict)
            # start_epoch = checkpoint['train_epoch']
            logging.info("Model loaded \n {}\n{}\n{}\n{}\n{}".format(discriminator,encoder,decoder,en_de_optimizer,cr_optimizer))
        # else:
            # start_epoch = 0
        logging.info("weight_e:{}".format(WEIGHT_E))
        logging.info("weight_d:{}".format(WEIGHT_D))
        logging.info("weight_a:{}".format(WEIGHT_A))
        start_epoch = 1
        if cuda_count >= 1:
            logging.info("cuda count is {} \ncuda count: {}".format(cuda_count,os.environ['CUDA_VISIBLE_DEVICES']))
            discriminator = nn.DataParallel(discriminator,device_ids=device_ids)
            encoder = nn.DataParallel(encoder,device_ids=device_ids)
            decoder = nn.DataParallel(decoder,device_ids=device_ids)
            
        de_optimizer = Adam(decoder.parameters(), lr=5e-4)
        attack_number = i
        for ep in range(start_epoch,epochs):
            metrics['train.encoder_mse'] = []
            metrics['train.decoder_loss'] = []
            metrics['train.decoder_acc'] = []
            metrics['val.encoder_mse'] = []
            metrics['val.decoder_loss'] = []
            metrics['val.decoder_acc'] = []
            logging.info('Epoch {}/{}'.format(ep, epochs))
            step = 1
            encoder.train()
            decoder.train()
            discriminator.train()
            for data in track(train_loader):
                cover = data[0]
                cover_phase = data[1].to(device)
                cover = cover.to(device)
                N, H, W = cover.size()
                payload = torch.zeros((N, 1, payload_size, payload_size),
                                      device=device).random_(0, 2)
                payload = (payload-0.5)*2
                generated = encoder.forward(cover, payload)
                #logging.info("generated.shape:{}".format(generated.shape))
                d_target_label_cover = torch.full((cover.shape[0], 1), cover_label, device=device).float()
                d_target_label_encoded = torch.full((cover.shape[0], 1), encoded_label, device=device).float()
                g_target_label_encoded = torch.full((cover.shape[0], 1), cover_label, device=device).float()


                # ---------------- Train the discriminator -----------------------------
                cr_optimizer.zero_grad()
                # train on cover
                cover_score = discriminator.forward(cover)
                d_loss_on_cover = bce_with_logits_loss(cover_score, d_target_label_cover)
                d_loss_on_cover.backward()
                # train on fake
                generated_score = discriminator.forward(generated.detach())
                d_loss_on_encoded = bce_with_logits_loss(generated_score, d_target_label_encoded)
                d_loss_on_encoded.backward()
                cr_optimizer.step()

                # --------------Train the generator (encoder-decoder) ---------------------
                en_de_optimizer.zero_grad()
                encoder_mse = mse_loss(generated, cover)
                y = dwt_inverse(generated, cover_phase)
                attacked_data = compress_attack(y, True,step,attack_net,temp_path)
                carrier_reconst_tag, _ = dwt_transform(attacked_data)
                decoded = decoder.forward(carrier_reconst_tag)
                decoder_loss = mse_loss(decoded, payload)
                decoder_acc = (decoded >= 0).eq(
                    payload >= 0).sum().float() / payload.numel()
                metrics['train.encoder_mse'].append(encoder_mse.item())
                metrics['train.decoder_loss'].append(decoder_loss.item())
                metrics['train.decoder_acc'].append(decoder_acc.item())
                generated_score = discriminator.forward(generated)
                g_loss_adv = bce_with_logits_loss(generated_score, g_target_label_encoded)
                (WEIGHT_E*encoder_mse + WEIGHT_D*decoder_loss + WEIGHT_A*g_loss_adv).backward()
                en_de_optimizer.step()
                if step % print_each == 0 or step == train_steps_in_epoch:
                    decoded_identity = decoder.forward(generated.detach())
                    decoder_acc_identity = (decoded_identity >= 0).eq(
                    payload >= 0).sum().float() / payload.numel()
                    logging.info('encoder_mse: {:.8f} - decoder_loss: {:.8f} - decoder_acc: {:.8f}/{:.8f} - d_c_adv_loss: {:.8f} - d_g_adv_loss: {:.8f} - g_adv_loss: {:.8f}'.format\
                        (encoder_mse.item(), decoder_loss.item(), decoder_acc.item(), decoder_acc_identity.item(), d_loss_on_cover.item(), d_loss_on_encoded.item(), g_loss_adv.item()))
                    logging.info('-' * 100)
                    save_model(encoder, decoder, en_de_optimizer, de_optimizer, discriminator,
                        cr_optimizer, metrics, ep, str(attack_number), str(decoder_acc), str(ep))
                step += 1

            if ep%1==0:
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    for data in track(valid_loader):
                        cover = data[0]
                        cover_phase = data[1].to(device)
                        cover = cover.to(device)
                        N, H, W = cover.size()
                        payload = torch.zeros((N, 1, payload_size, payload_size),
                                            device=device).random_(0, 2)
                        payload = (payload-0.5)*2
                        generated = encoder.forward(cover, payload)
                        y = dwt_inverse(generated, cover_phase)
                        attacked_data = compress_attack(y, False,step,attack_net,temp_path)
                        carrier_reconst_tag, _ = dwt_transform(attacked_data)
                        decoded = decoder.forward(carrier_reconst_tag)
                        encoder_mse = mse_loss(generated, cover)
                        decoder_loss = mse_loss(decoded, payload)
                        decoder_acc = (decoded >= 0).eq(
                            payload >= 0).sum().float() / payload.numel()
                        cover_aud_tensor = dwt_inverse(cover, cover_phase)
                        metrics['val.encoder_mse'].append(encoder_mse.item())
                        metrics['val.decoder_loss'].append(decoder_loss.item())
                        metrics['val.decoder_acc'].append(decoder_acc.item())

                avg_acc = np.mean(metrics['val.decoder_acc'])
                logging.info('\nencoder_mse: {:.8f} - decoder_loss: {:.8f} - average_acc: {:.8f} - d_c_adv_loss: {:.8f} - d_g_adv_loss: {:.8f} - g_adv_loss: {:.8f}'.format\
                            (encoder_mse.item(), decoder_loss.item(), avg_acc.item(), d_loss_on_cover.item(), d_loss_on_encoded.item(), g_loss_adv.item()))
                # if ep >= 80 or ep%save_circle == 0:
                save_model(encoder, decoder, en_de_optimizer, de_optimizer, discriminator,
                            cr_optimizer, metrics, ep, str(attack_number), str(avg_acc), str(ep))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Watermark Model')
    parser.add_argument('-lr_de', '--lr_de', type=float,
                        default=1e-4, help='de_optimizer learning rate')
    parser.add_argument('-lr_en_de', '--lr_en_de', type=float,
                        default=1e-4, help='en_de_optimizer learning rate')
    parser.add_argument('-lr_cr', '--lr_cr', type=float,
                        default=1e-4, help='cr_optimizer learning rate')
    args = parser.parse_args()
    logging.info("*"*10 + "Train Audio DWT" + "*"*10)
    torch.multiprocessing.set_start_method('spawn')

    for func in [
        lambda: os.mkdir(os.path.join('.', 'results')),
        lambda: os.mkdir(os.path.join('.', SAVE_MODEL_PATH))]:
        try:
            func()
        except Exception as error:
            print(error)
            continue

    METRIC_FIELDS = [
        'val.encoder_mse',
        'val.decoder_loss',
        'val.decoder_acc',
        'train.encoder_mse',
        'train.decoder_loss',
        'train.decoder_acc',
        'train.attacker_mse',
    ]

    temp_path="/public/jiangyichuan/DeAR/music/{}".format(datetime.datetime.now())
    os.mkdir(temp_path)
    for i in range(10):
        os.mkdir("{}/{}".format(temp_path,i))

    
    data_dir = data_dir
    channels_size = CHANNEL_SIZE
    transform = transforms.Compose([my_transform()])
    train_set = my_dataset(os.path.join(data_dir, "train/"))
    part_train_set = torch.utils.data.random_split(train_set, [TRAIN_DATA, len(train_set)-TRAIN_DATA])[0]
    train_loader = torch.utils.data.DataLoader(part_train_set, batch_size=BATCH, shuffle=True, num_workers=num_workers)
    valid_set = my_dataset(os.path.join(data_dir, "val/"))
    part_valid_set = torch.utils.data.random_split(valid_set, [VAL_DATA, len(valid_set)-VAL_DATA])[0]
    valid_loader = torch.utils.data.DataLoader(part_valid_set, batch_size=BATCH, shuffle=False, num_workers=num_workers)
    
    encoder = Encoder(
        data_depth, hidden_size, channels_size)
    decoder = Decoder(
        data_depth, hidden_size, channels_size)
    discriminator = Discriminator(data_depth, hidden_size, channels_size)

    

    logging.info("#"*80 + "encoder:\n{}\n".format(encoder)+"#"*80 + "decoder:\n{}\n".format(decoder)+"#"*80 + "discriminator:\n{}\n".format(discriminator))
    if cuda_count > 1 and pre_epochs>0:
        encoder=nn.DataParallel(encoder)
        decoder=nn.DataParallel(decoder)
        discriminator=nn.DataParallel(discriminator)
        encoder.to(device)
        decoder.to(device)
        discriminator.to(device)
        
    attack_net=unet.Model()    
    attack_net=nn.DataParallel(attack_net)
    attack_net.load_state_dict(torch.load("/public/jiangyichuan/DeAR/DeAR_source_code/distortion_layer/unet/epoch_67_avg_loss_877.34228515625_wav_aac_wav")["state_dict_net"])
    attack_net.to(device)


    cr_optimizer = Adam(discriminator.parameters(), lr=args.lr_cr)
    en_de_optimizer = Adam([
	{'params': decoder.parameters(), 'lr': args.lr_en_de}, 
	{'params': encoder.parameters(), 'lr': args.lr_en_de}
	])
    de_optimizer = Adam(decoder.parameters(), lr=args.lr_de)

    scheduler = lr_scheduler.StepLR(en_de_optimizer, step_size=25, gamma=0.5)
    metrics = {field: list() for field in METRIC_FIELDS}

    train_model(scheduler, encoder, decoder, en_de_optimizer, de_optimizer,\
                    discriminator, cr_optimizer, metrics, train_loader, valid_loader)
