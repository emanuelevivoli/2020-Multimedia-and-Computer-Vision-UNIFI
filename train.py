import argparse
import os
from math import log10

import lpips
# from lpips_pytorch import LPIPS, lpips

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, scalePixels, weight_init
from loss import GeneratorLoss
import torch.nn as nn
from model import Generator, VGGStyleDiscriminator128 # , Discriminator

from datetime import datetime

parser = argparse.ArgumentParser(description='Train Compression Artifact Removal Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8], help='re-sizing upscale factor')
parser.add_argument('--quality_factor', default=20, type=int, choices=[10, 20, 30, 40], help='dagrading quality factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--loss_choice', default='CUSTOM', type=str, choices=['CUSTOM', 'BCE'], help='type of loss')
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
    LOSS = opt.loss_choice

    loss_fn_alex = lpips.LPIPS(net='alex', spatial=True) # best forward scores
    # lpips_criterion = LPIPS(
    #     net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
    #     version='0.1'  # Currently, v0.1 is supported
    # )

    # weight_path_netD = 'results_CRGAN/epochs/run15-12-2020_23:34:50_crop128_batch64_upscale2_qf20_epoch5_netD.pth'

    torch.autograd.set_detect_anomaly(True)
    
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR, crop_numb=100)
    # val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR)
    val_set = TrainDatasetFromFolder('data/DIV2K_valid_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, quality_factor=QUALITY_FACTOR, train=False)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = Generator(QUALITY_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    
    netD = VGGStyleDiscriminator128() # Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # ! load weights for Discriminator
    # netD.load_state_dict(torch.load(weight_path_netD))

    # ! xavier inizialization
    weight_init(netG)
    weight_init(netD)

    generator_criterion = GeneratorLoss()
    

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
        loss_fn_alex.cuda()

    if LOSS == 'BCE':
        discriminator_criterion = nn.BCELoss()
        if torch.cuda.is_available():
            discriminator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = { 'TRAIN_d_loss': [], 'TRAIN_g_loss': [],
                'TRAIN_d_score': [], 'TRAIN_g_score': [],

                'VAL_d_loss': [],
                'VAL_d_score': [], 'VAL_g_score': [],

                'jpeg_loss': [], 'gen_loss': [], 'vgg_loss': [], 'mse_loss': [], 'tv_loss': [],

                #, 'psnr': [], 'ssim': []
                }

    pre_path = 'results_' + MODEL_NAME + '/'

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)

        losses_results = {'jpeg_loss': 0, 'gen_loss': 0, 'vgg_loss': 0, 'mse_loss': 0, 'tv_loss': 0}
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for jpeg, target in train_bar:

            g_update_first = True
            batch_size = target.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            
            # real_img = Variable(target)
            # if torch.cuda.is_available():
            #     real_img = real_img.cuda()
            # z = Variable(target)
            # if torch.cuda.is_available():
            #     z = z.cuda()
            # if SKIP_GEN:
            #     if torch.cuda.is_available():
            #         jpeg = jpeg.cuda()
            #     fake_img = Variable(jpeg)
            # else:
            #     fake_img = netG(z)

            optimizerD.zero_grad()
            

            # real_out = netD(real_img).mean()
            # fake_out = netD(fake_img).mean()

            real_img = Variable(target)
            fake_img = Variable(target)

            if torch.cuda.is_available():
                real_img = real_img.cuda()
                fake_img = fake_img.cuda()

            real_out = netD(real_img)
            fake_img = netG(fake_img)
            fake_out = netD(fake_img)

            if LOSS == 'BCE':
                output = torch.cat((real_out, fake_out), 0).view(-1)
                labels_ones = torch.ones([batch_size, 1], dtype=torch.float32)
                labels_zero = torch.zeros([batch_size, 1], dtype=torch.float32)
                labels = torch.cat((labels_ones, labels_zero), 0).view(-1)
                if torch.cuda.is_available():
                    output = output.cuda()
                    labels = labels.cuda()
                d_loss = discriminator_criterion(output, labels)
            else:
                d_loss = - (torch.log(real_out.mean()) + torch.log(1 - fake_out.mean()))

            # print(d_loss.size())
            d_loss.backward(retain_graph=True)
            
            # ? d_loss = torch.log(1 - real_out + fake_out)
            # d_loss = 1 - real_out + fake_out
            # d_loss = - (torch.log(real_out) + torch.log(1 - fake_out))
            # d_loss.backward(retain_graph=True)

            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            optimizerG.zero_grad()

            jpeg_loss, mse_loss, gen_loss, vgg_loss, tv_loss = generator_criterion(fake_out, fake_img, real_img, QUALITY_FACTOR)
            # g_loss = gen_loss * 0.001 + vgg_loss + mse_loss + tv_loss # + jpeg_loss

            # print(jpeg_loss.size())
            # print(mse_loss.size())
            # print(gen_loss.size())
            # print(vgg_loss.size())
            # print(tv_loss.size())

            g_loss = gen_loss * 0.001 + vgg_loss + mse_loss + tv_loss + jpeg_loss
            # print(g_loss.size())
            g_loss.backward()
            
            # fake_img = netG(fake_img)
            # fake_out = netD(fake_img).mean()
            
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
            running_results['d_score'] += real_out.mean().item() * batch_size
            running_results['g_score'] += fake_out.mean().item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] (TRAIN) Loss [ D | G ]: [%.4f | %.4f] - Score [D(x) | D(G(x))]: [%.4f | %.4f] ' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        
        file_name = 'run'+str(DAY_TIME) + '_crop'+str(CROP_SIZE) + '_batch'+str(BATCH_SIZE) + '_upscale'+str(UPSCALE_FACTOR) + '_qf'+str(QUALITY_FACTOR) + '_epochs'+str(NUM_EPOCHS)
        out_path = pre_path + 'val_results/' + file_name + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        netG.eval()
        netD.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            # valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            valing_scores = {'batch_sizes': 0, 'd_loss': 0, 'd_score': 0, 'g_score': 0}
            valing_results = {'batch_sizes': 0, 'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'lpips': 0}
            val_images = []

            for val_jr, val_hr in val_bar:

                batch_size = val_jr.size(0)
                valing_scores['batch_sizes'] += batch_size
                valing_results['batch_sizes'] += batch_size

                val_real_img = Variable(val_hr)
                val_fake_img = Variable(val_hr)

                if torch.cuda.is_available():
                    val_real_img = val_real_img.cuda()
                    val_fake_img = val_fake_img.cuda()
                
                val_real_out = netD(val_real_img)
                val_fake_img = netG(val_fake_img)
                val_fake_out = netD(val_fake_img)

                if LOSS == 'BCE':
                    val_output = torch.cat((val_real_out, val_fake_out), 0).view(-1)
                    val_labels_ones = torch.ones([batch_size, 1], dtype=torch.float32)
                    val_labels_zero = torch.zeros([batch_size, 1], dtype=torch.float32)
                    val_labels = torch.cat((val_labels_ones, val_labels_zero), 0).view(-1)
                    if torch.cuda.is_available():
                        val_output = val_output.cuda()
                        val_labels = val_labels.cuda()
                    val_d_loss = discriminator_criterion(val_output, val_labels)
                else:
                    val_d_loss = - (torch.log(val_real_out) + torch.log(1 - val_fake_out))

                # loss for current batch before optimization 
                valing_scores['d_loss'] += val_d_loss.item() * batch_size

                valing_scores['d_score'] += (val_real_out).mean().item() * batch_size
                valing_scores['g_score'] += (val_fake_out).mean().item() * batch_size
                
                batch_mse = ((val_fake_img - val_real_img) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(val_fake_img, val_real_img).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((val_real_img.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                
                # print(val_real_img.size())
                # print(val_fake_img.size())
                
                # lpips_real_img = lpips.im2tensor(val_real_img)
                # lpips_fake_img = lpips.im2tensor(val_fake_img)

                # ! scale
                scaled_real_img = scalePixels(val_real_img)
                scaled_fake_img = scalePixels(val_fake_img)
                valing_results['lpips'] = loss_fn_alex.forward(scaled_real_img, scaled_fake_img).squeeze().mean()
                
                # ? no scale
                # ? valing_results['lpips'] = loss_fn_alex(val_real_img, val_fake_img)


                # valing_results['lpips'] = lpips_criterion(val_real_img, val_fake_img)

                # print(valing_results['lpips'])
                # print(valing_results['lpips'].size())
                
                # val_bar.set_description(
                #     desc='[Validation] Jpeg: [%d|%d|%d|%d] Real: [%d|%d|%d|%d] Fake: [%d|%d|%d|%d]' % (
                #         val_jr.size(0), val_jr.size(1), val_jr.size(2), val_jr.size(3),
                #         val_hr.size(0), val_hr.size(1), val_hr.size(2), val_hr.size(3),
                #         jrr.size(0), jrr.size(1), jrr.size(2), jrr.size(3),
                #         ))

                val_bar.set_description(desc='[%d/%d] (VAL)   Loss [ D | G ]: [%.4f |  ???  ] - Score [D(x) | D(G(x))]: [%.4f | %.4f] - SSIM|PSNR|LPIPS: [%.4f | %.4f | %.4f]' % (
                    epoch, NUM_EPOCHS, 
                    valing_scores['d_loss'] / valing_scores['batch_sizes'],
                    valing_scores['d_score'] / valing_scores['batch_sizes'],
                    valing_scores['g_score'] / valing_scores['batch_sizes'],
                    valing_results['ssim'] / valing_results['batch_sizes'],
                    valing_results['psnr'] / valing_results['batch_sizes'],
                    valing_results['lpips'] / valing_results['batch_sizes']))

                val_images.extend(
                    [display_transform()(val_jr.squeeze(0)), display_transform()(val_hr.squeeze(0)),
                     display_transform()(val_fake_img.data.cpu().squeeze(0))])

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    
        # save model parameters
        # torch.save(netG.state_dict(), pre_path + 'epochs/run%s_crop%d_batch%d_upscale%d_qf%d_epoch%d_netG.pth' % (DAY_TIME, CROP_SIZE, BATCH_SIZE, UPSCALE_FACTOR, QUALITY_FACTOR, epoch))
        # torch.save(netD.state_dict(), pre_path + 'epochs/run%s_crop%d_batch%d_upscale%d_qf%d_epoch%d_netD.pth' % (DAY_TIME, CROP_SIZE, BATCH_SIZE, UPSCALE_FACTOR, QUALITY_FACTOR, epoch))


        # save loss\scores\psnr\ssim
        results['TRAIN_d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['TRAIN_g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['TRAIN_d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['TRAIN_g_score'].append(running_results['g_score'] / running_results['batch_sizes'])

        results['VAL_d_loss'].append(valing_scores['d_loss'] / valing_scores['batch_sizes'])
        results['VAL_d_score'].append(valing_scores['d_score'] / valing_scores['batch_sizes'])
        results['VAL_g_score'].append(valing_scores['g_score'] / valing_scores['batch_sizes'])

        results['jpeg_loss'].append(losses_results['jpeg_loss'] / running_results['batch_sizes'])
        results['gen_loss'].append(losses_results['gen_loss'] / running_results['batch_sizes'])
        results['vgg_loss'].append(losses_results['vgg_loss'] / running_results['batch_sizes'])
        results['mse_loss'].append(losses_results['mse_loss'] / running_results['batch_sizes'])
        results['tv_loss'].append(losses_results['tv_loss'] / running_results['batch_sizes'])

        # results['psnr'].append(valing_results['psnr'])
        # results['ssim'].append(valing_results['ssim'])
    
        if epoch % 2 == 0 and epoch != 0:
            out_path = pre_path + 'statistics/'
            data_frame = pd.DataFrame(
                data={
                    'TRAIN_Loss_D': results['TRAIN_d_loss'], 
                    'TRAIN_Loss_G': results['TRAIN_g_loss'], 
                    'TRAIN_Score_D': results['TRAIN_d_score'], 
                    'TRAIN_Score_G': results['TRAIN_g_score'], 

                    'VAL_Loss_D': results['VAL_d_loss'], 
                    'VAL_Score_D': results['VAL_d_score'], 
                    'VAL_Score_G': results['VAL_g_score'], 
                    
                    'Loss_JPEG': results['jpeg_loss'], 'Loss_GEN': results['gen_loss'], 'Loss_VGG': results['vgg_loss'], 'Loss_MSE': results['mse_loss'], 'Loss_TV': results['tv_loss'],
                    # 'PSNR': results['psnr'], 'SSIM': results['ssim']
                },
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + file_name + '_train_results.csv', index_label='Epoch')
