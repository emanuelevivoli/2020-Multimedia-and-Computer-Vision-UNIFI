import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from model import VGGStyleDiscriminator128

from datetime import datetime

parser = argparse.ArgumentParser(description='Train Discriminator to distinguish Jpeg vs. Real Images')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8], help='re-sizing upscale factor')
parser.add_argument('--quality_factor', default=20, type=int, choices=[10, 20, 30, 40], help='dagrading quality factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')


if __name__ == '__main__':
    opt = parser.parse_args()
    now = datetime.now()

    DAY_TIME = now.strftime("%d-%m-%Y_%H:%M:%S")
    MODEL_NAME = 'JPEG_CLASSIFICATION'
    
    CROP_SIZE = opt.crop_size
    BATCH_SIZE = opt.batch_size
    UPSCALE_FACTOR = opt.upscale_factor
    QUALITY_FACTOR = opt.quality_factor
    NUM_EPOCHS = opt.num_epochs

    torch.autograd.set_detect_anomaly(True)
    
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR, crop_numb=100)
    # val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR)
    val_set = TrainDatasetFromFolder('data/DIV2K_valid_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netD = VGGStyleDiscriminator128(num_out=1, sigmoid_out=False)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    discriminator_criterion = nn.BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        netD.cuda()
        discriminator_criterion.cuda()
    
    discriminator_optimizer = optim.Adam(netD.parameters())
    
    results = { 'TRAIN_d_loss': [], 'TRAIN_real_score': [], 'TRAIN_jpeg_score': [],
                'VAL_d_loss': [], 'VAL_real_score': [], 'VAL_jpeg_score': [] }
    
    pre_path = 'results_' + MODEL_NAME + '/'
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)

        running_results = {'batch_sizes': 0, 'd_loss': 0, 'real_score': 0, 'jpeg_score': 0}

        # train_images = []

        netD.train()
        for jpeg, target in train_bar:

            batch_size = target.size(0)
            running_results['batch_sizes'] += batch_size
    
            discriminator_optimizer.zero_grad()
            
            ############################
            # (1) D network
            ###########################
            
            real_img = Variable(target)
            fake_img = Variable(jpeg)
                        
            if torch.cuda.is_available():
                real_img = real_img.cuda()
                fake_img = fake_img.cuda()
            
            real_out = netD(real_img)
            fake_out = netD(fake_img)
            
            output = torch.cat((real_out, fake_out), 0).view(-1)

            labels_ones = torch.ones([batch_size, 1], dtype=torch.float32)
            labels_zero = torch.zeros([batch_size, 1], dtype=torch.float32)
            labels = torch.cat((labels_ones, labels_zero), 0).view(-1)

            if torch.cuda.is_available():
                output = output.cuda()
                labels = labels.cuda()
            
            d_loss = discriminator_criterion(output, labels)
            d_loss.backward()
            discriminator_optimizer.step()
            
            # loss for current batch before optimization 
            running_results['d_loss'] += d_loss.item() * batch_size
            
            # print(real_out)
            # print(fake_out)
            # print(output)
            # print(labels)

            running_results['real_score'] += torch.sigmoid(real_out).mean().item() * batch_size
            running_results['jpeg_score'] += torch.sigmoid(fake_out).mean().item() * batch_size
            
            # train_images.extend([display_transform()(real_img[0].data.cpu().squeeze(0)), display_transform()(fake_img[0].data.cpu().squeeze(0))])

            train_bar.set_description(desc='[%d/%d] (TRAIN) Loss_D: %.4f D(x): %.4f D(Jpeg(x)): %.4f' % (
                epoch, NUM_EPOCHS, 
                running_results['d_loss'] / running_results['batch_sizes'],
                running_results['real_score'] / running_results['batch_sizes'],
                running_results['jpeg_score'] / running_results['batch_sizes']))
    
        # create folders for save data/images (train)
        file_name = 'run'+str(DAY_TIME) + '_crop'+str(CROP_SIZE) + '_batch'+str(BATCH_SIZE) + '_upscale'+str(UPSCALE_FACTOR) + '_qf'+str(QUALITY_FACTOR) + '_epochs'+str(NUM_EPOCHS)
        train_out_path = pre_path + 'train_results/' + file_name + '/'
        if not os.path.exists(train_out_path):
            os.makedirs(train_out_path)
        
        # ? save images (train)
        # train_images = torch.stack(train_images)
        # train_images = torch.chunk(train_images, train_images.size(0) // 15)
        # val_save_bar = tqdm(train_images, desc='[saving training results]')
        # index = 1
        # for image in val_save_bar:
        #     image = utils.make_grid(image, nrow=4, padding=5)
        #     utils.save_image(image, train_out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        #     index += 1

        # ? create folders for save data/images (val)
        # val_out_path = pre_path + 'val_results/' + file_name + '/'
        # if not os.path.exists(val_out_path):
        #     os.makedirs(val_out_path)
        
        netD.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'batch_sizes': 0, 'd_loss': 0, 'real_score': 0, 'jpeg_score': 0}
            # val_images = []

            for val_jr, val_hr in val_bar:

                batch_size = val_jr.size(0)
                valing_results['batch_sizes'] += batch_size
                
                val_real_img = Variable(val_hr)
                val_fake_img = Variable(val_jr)
                            
                if torch.cuda.is_available():
                    val_real_img = val_real_img.cuda()
                    val_fake_img = val_fake_img.cuda()
                
                val_real_out = netD(val_real_img)
                val_fake_out = netD(val_fake_img)
                
                val_output = torch.cat((val_real_out, val_fake_out), 0).view(-1)

                val_labels_ones = torch.ones([batch_size, 1], dtype=torch.float32)
                val_labels_zero = torch.zeros([batch_size, 1], dtype=torch.float32)
                val_labels = torch.cat((val_labels_ones, val_labels_zero), 0).view(-1)

                if torch.cuda.is_available():
                    val_output = val_output.cuda()
                    val_labels = val_labels.cuda()
                
                val_d_loss = discriminator_criterion(val_output, val_labels)
                
                # loss for current batch before optimization 
                valing_results['d_loss'] += val_d_loss.item() * batch_size

                valing_results['real_score'] += torch.sigmoid(val_real_out).mean().item() * batch_size
                valing_results['jpeg_score'] += torch.sigmoid(val_fake_out).mean().item() * batch_size
                
                # val_images.extend([display_transform()(val_real_img[0].data.cpu().squeeze(0)), display_transform()(val_fake_img[0].data.cpu().squeeze(0))])

                val_bar.set_description(desc='[%d/%d] (VAL) Loss_D: %.4f D(x): %.4f D(Jpeg(x)): %.4f' % (
                    epoch, NUM_EPOCHS, 
                    valing_results['d_loss'] / valing_results['batch_sizes'],
                    valing_results['real_score'] / valing_results['batch_sizes'],
                    valing_results['jpeg_score'] / valing_results['batch_sizes']))

            # val_images = torch.stack(val_images)
            # val_images = torch.chunk(val_images, val_images.size(0) // 15)
            # val_save_bar = tqdm(val_images, desc='[saving validation results]')
            # index = 1
            # for image in val_save_bar:
            #     image = utils.make_grid(image, nrow=4, padding=5)
            #     utils.save_image(image, val_out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            #     index += 1
    
        # ? save model parameters
        torch.save(netD.state_dict(), pre_path + 'epochs/run%s_crop%d_batch%d_upscale%d_qf%d_epoch%d_netD.pth' % (DAY_TIME, CROP_SIZE, BATCH_SIZE, UPSCALE_FACTOR, QUALITY_FACTOR, epoch))
        
        results['TRAIN_d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['TRAIN_real_score'].append(running_results['real_score'] / running_results['batch_sizes'])
        results['TRAIN_jpeg_score'].append(running_results['jpeg_score'] / running_results['batch_sizes'])

        results['VAL_d_loss'].append(valing_results['d_loss'] / valing_results['batch_sizes'])
        results['VAL_real_score'].append(valing_results['real_score'] / valing_results['batch_sizes'])
        results['VAL_jpeg_score'].append(valing_results['jpeg_score'] / valing_results['batch_sizes'])

        # save loss\scores
        if epoch % 2 == 0 and epoch != 0:
            out_path = pre_path + 'statistics/'
            data_frame = pd.DataFrame(
                data={
                    'TRAIN_Loss_D': results['TRAIN_d_loss'],
                    'TRAIN_Score_D': results['TRAIN_real_score'], 
                    'TRAIN_Score_G': results['TRAIN_jpeg_score'],
                    'VAL_Loss_D': results['VAL_d_loss'],
                    'VAL_Score_D': results['VAL_real_score'], 
                    'VAL_Score_G': results['VAL_jpeg_score']
                },
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + file_name + '_train_results.csv', index_label='Epoch')
