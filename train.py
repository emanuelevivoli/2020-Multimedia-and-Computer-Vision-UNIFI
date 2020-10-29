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

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

#########################
# Show image inline
#########################
# ! import matplotlib.pyplot as plt
# ! import matplotlib.image as mpimg
# ! import numpy as np
#########################

parser = argparse.ArgumentParser(description='Train Compression Artifact Removal Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8], help='re-sizing upscale factor')
parser.add_argument('--quality_factor', default=20, type=int, choices=[10, 20, 30, 40], help='dagrading quality factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')


if __name__ == '__main__':
    opt = parser.parse_args()
    
    MODEL_NAME = 'CRGAN'
    CROP_SIZE = opt.crop_size
    BATCH_SIZE = opt.batch_size
    UPSCALE_FACTOR = opt.upscale_factor
    QUALITY_FACTOR = opt.quality_factor
    NUM_EPOCHS = opt.num_epochs

    torch.autograd.set_detect_anomaly(True)
    
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = Generator(QUALITY_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    pre_path = 'results_' + MODEL_NAME + '/'

    # ! jpeg_folder_images = 'JPEG_examples/'
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        # to distinguish
        # ! jpeg_number = 0 

        netG.train()
        netD.train()
        for jpeg, target in train_bar:

            # torch.Size([8, 3, 44, 44])
            # print('#size _:', _.size()) 

            # torch.Size([8, 3, 88, 88])
            # print('#size target:', target.size())

            #########################
            # Show jpeg images batch
            # ! temp_jpeg = np.transpose(jpeg[0, :, :, :], (1,2,0))
            # ! imgplot = plt.imshow(temp_jpeg)
            # ! plt.savefig(jpeg_folder_images + 'train/' + str(jpeg_number) + '.jpg')
            #########################

            g_update_first = True
            batch_size = target.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            # we need to execute jpeg compr. on original targer
            # z = Variable(_) 
            z = Variable(target)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)
            # ! fake_img = netG(z, jpeg_number)

            # torch.Size([8, 3, 88, 88])
            # print('#size z:', z.size()) 

            # torch.Size([8, 3, 88, 88])
            # print('#size netG(z):', fake_img.size())

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img, QUALITY_FACTOR)
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            
            
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
            # ! jpeg_number += 1

        netG.eval()
        out_path = pre_path + 'val_results/' + 'crop'+str(CROP_SIZE) + '_batch'+str(BATCH_SIZE) + '_upscale'+str(UPSCALE_FACTOR) + '_qf'+str(QUALITY_FACTOR) + '_epochs'+str(NUM_EPOCHS) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []

            # to distinguish
            # ! jpeg_number = 0 

            for val_jr, val_hr in val_bar:
                batch_size = val_jr.size(0)
                valing_results['batch_sizes'] += batch_size
                hr = val_hr
                if torch.cuda.is_available():
                    hr = hr.cuda()
                jrr = netG(hr)
        
                batch_mse = ((jrr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(jrr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to CR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                #########################
                # Show jpeg images batch
                # ! temp_jpeg = np.transpose(val_jr[0, :, :, :], (1,2,0))
                # ! imgplot = plt.imshow(temp_jpeg)
                # ! plt.savefig(jpeg_folder_images + 'val/' + str(jpeg_number) + '.jpg')
                #########################

                val_images.extend(
                    [display_transform()(val_jr.squeeze(0)), display_transform()(val_hr.squeeze(0)),
                     display_transform()(jrr.data.cpu().squeeze(0))])
                
                # ! jpeg_number += 1

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    
        # save model parameters
        torch.save(netG.state_dict(), pre_path + 'epochs/netG_crop%d_batch%d_upscale%d_qf%d_epoch%d.pth' % (CROP_SIZE, BATCH_SIZE, UPSCALE_FACTOR, QUALITY_FACTOR, epoch))
        torch.save(netD.state_dict(), pre_path + 'epochs/netD_crop%d_batch%d_upscale%d_qf%d_epoch%d.pth' % (CROP_SIZE, BATCH_SIZE, UPSCALE_FACTOR, QUALITY_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = pre_path + 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'crop%d_batch%d_upscale%d_qf%d_epochs%d' % (CROP_SIZE, BATCH_SIZE, UPSCALE_FACTOR, QUALITY_FACTOR, NUM_EPOCHS) + '_train_results.csv', index_label='Epoch')
