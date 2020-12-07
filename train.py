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
from model import Generator, VGGStyleDiscriminator128 # , Discriminator

from datetime import datetime

# from utils.noise_image import get_white_noise_tensor

#########################
# Show image inline
#########################
# ! import matplotlib.pyplot as plt
# ! import matplotlib.image as mpimg
# ! import numpy as np
#########################

parser = argparse.ArgumentParser(description='Train Compression Artifact Removal Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8], help='re-sizing upscale factor')
parser.add_argument('--quality_factor', default=20, type=int, choices=[10, 20, 30, 40], help='dagrading quality factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
# parser.add_argument('--fake', default=False, type=bool, choices=[True, False], help='Random input to generator if fake = True')
parser.add_argument('--skip_gen', default=False, type=bool, choices=[True, False], help='Jpeg input for Discriminator (no uses of Generator at all) if skip_gen = True')


if __name__ == '__main__':
    opt = parser.parse_args()
    
    now = datetime.now()
    DAY_TIME = now.strftime("%d-%m-%Y_%H:%M:%S")
    MODEL_NAME = 'CRGAN'
    CROP_SIZE = opt.crop_size
    BATCH_SIZE = opt.batch_size
    UPSCALE_FACTOR = opt.upscale_factor
    QUALITY_FACTOR = opt.quality_factor
    NUM_EPOCHS = opt.num_epochs
    SKIP_GEN = opt.skip_gen

    torch.autograd.set_detect_anomaly(True)
    
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = Generator(QUALITY_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # netD = Discriminator()
    netD = VGGStyleDiscriminator128()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = { 'd_loss': [], 'g_loss': [], 
                'jpeg_loss': [], 'gen_loss': [], 'vgg_loss': [], 'mse_loss': [], 'tv_loss': [],
                'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    
    pre_path = 'results_' + MODEL_NAME + '/'
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)

        losses_results = {'jpeg_loss': 0, 'gen_loss': 0, 'vgg_loss': 0, 'mse_loss': 0, 'tv_loss': 0}
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for jpeg, target in train_bar:
            # ? jpeg, target are tensor [batch_size, channels, height, length]

            g_update_first = True
            batch_size = target.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            
            # ? real_img.size = [batch_size, channels, height, length]
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            
            # ? z.size = [batch_size, channels, height, length]
          
            z = Variable(target)

            if torch.cuda.is_available():
                z = z.cuda()

            # ? fake_img.size = [batch_size, channels, height, length]
            if SKIP_GEN:
                
                if torch.cuda.is_available():
                    jpeg = jpeg.cuda()

                fake_img = Variable(jpeg)
                
            else:
                fake_img = netG(z)

            netD.zero_grad()

            # ! SCORE_D (real_out) e SCORE_G (fake_out)

            # ? netD(real_img).size = [batch_size, 1]
            # ? netD(real_img).mean().size = [1]
            real_out = netD(real_img).mean()

            # ? netD(fake_img).size = [batch_size, 1]
            # ? netD(fake_img).mean().size = [1]
            fake_out = netD(fake_img).mean()
            
            # ! they must be different (real_out + fake_out)
            
            # ? d_loss = torch.log(1 - real_out + fake_out)
            # ??? d_loss = 1 - real_out + fake_out
            d_loss = - (torch.log(real_out) + torch.log(1 - fake_out))
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            jpeg_loss, mse_loss, gen_loss, vgg_loss, tv_loss = generator_criterion(fake_out, fake_img, real_img, QUALITY_FACTOR)
            # g_loss = (0.5 * d_loss.item()) * gen_loss + (1 - 0.5 * d_loss.item()) * ( jpeg_loss + 0.001 * gen_loss + vgg_loss + mse_loss + tv_loss )
            # ??? g_loss = gen_loss * 0.001 + vgg_loss + mse_loss + tv_loss + jpeg_loss
            g_loss = gen_loss # * 0.001 + vgg_loss + mse_loss + tv_loss + jpeg_loss
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            
            optimizerG.step()

            # jpeg/gen/vgg/mse/tv losses for current bacth before optimization
            losses_results['jpeg_loss'] += jpeg_loss.item() * batch_size
            losses_results['gen_loss'] += gen_loss.item() * batch_size
            losses_results['vgg_loss'] += vgg_loss.item() * batch_size
            losses_results['mse_loss'] += mse_loss.item() * batch_size
            losses_results['tv_loss'] += tv_loss.item() * batch_size
        
            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            
            # ? D_SCORE (real_out) e G_SCORE (fake_out)
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        file_name = 'run'+str(DAY_TIME) + '_crop'+str(CROP_SIZE) + '_batch'+str(BATCH_SIZE) + '_upscale'+str(UPSCALE_FACTOR) + '_qf'+str(QUALITY_FACTOR) + '_epochs'+str(NUM_EPOCHS)
        out_path = pre_path + 'val_results/' + file_name + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []

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
                val_images.extend(
                    [display_transform()(val_jr.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(jrr.data.cpu().squeeze(0))])

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    
        # save model parameters
        torch.save(netG.state_dict(), pre_path + 'epochs/run%s_crop%d_batch%d_upscale%d_qf%d_epoch%d_netG.pth' % (DAY_TIME, CROP_SIZE, BATCH_SIZE, UPSCALE_FACTOR, QUALITY_FACTOR, epoch))
        torch.save(netD.state_dict(), pre_path + 'epochs/run%s_crop%d_batch%d_upscale%d_qf%d_epoch%d_netD.pth' % (DAY_TIME, CROP_SIZE, BATCH_SIZE, UPSCALE_FACTOR, QUALITY_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])

        results['jpeg_loss'].append(losses_results['jpeg_loss'] / running_results['batch_sizes'])
        results['gen_loss'].append(losses_results['gen_loss'] / running_results['batch_sizes'])
        results['vgg_loss'].append(losses_results['vgg_loss'] / running_results['batch_sizes'])
        results['mse_loss'].append(losses_results['mse_loss'] / running_results['batch_sizes'])
        results['tv_loss'].append(losses_results['tv_loss'] / running_results['batch_sizes'])

        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 2 == 0 and epoch != 0:
            out_path = pre_path + 'statistics/'
            data_frame = pd.DataFrame(
                data={
                    'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 
                    'Score_D': results['d_score'], 'Score_G': results['g_score'], 
                    'Loss_JPEG': results['jpeg_loss'], 'Loss_GEN': results['gen_loss'], 'Loss_VGG': results['vgg_loss'], 'Loss_MSE': results['mse_loss'], 'Loss_TV': results['tv_loss'],
                    'PSNR': results['psnr'], 'SSIM': results['ssim']
                },
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + file_name + '_train_results.csv', index_label='Epoch')
