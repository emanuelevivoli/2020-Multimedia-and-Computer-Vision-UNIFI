import math
import torch
from torch import nn
from utils.jpeg_layer import jpegLayer

#########################
# Show image inline
#########################
# ! import matplotlib.pyplot as plt
# ! import matplotlib.image as mpimg
# ! import numpy as np
#########################


class Generator(nn.Module):
    def __init__(self, quality_factor):
        
        super(Generator, self).__init__()
        self.quality_factor = quality_factor

        self.jpeg = JpegComp()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.block8 = nn.Sequential(
            UpsampleBLock(64, 2),
            nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )

    def forward(self, x, jpeg_number=0):
        x = self.jpeg(x, self.quality_factor)
        # print('#Gen      x:', x.size()) 
        
        #########################
        # Show jpeg images batch
        # ! jpeg_folder_images = 'JPEG_examples/'
        # ! temp_jpeg = x
        # ! temp_jpeg = np.transpose(temp_jpeg.cpu()[0, :, :, :], (1,2,0))
        # ! imgplot = plt.imshow(temp_jpeg)
        # ! plt.savefig(jpeg_folder_images + 'forward/' + str(jpeg_number) + '.jpg')
        #########################
        
        block1 = self.block1(x)
        # print('#Gen block1:', block1.size()) 
        block2 = self.block2(block1)
        # print('#Gen block2:', block2.size()) 
        block3 = self.block3(block2)
        # print('#Gen block3:', block3.size())
        block4 = self.block4(block3)
        # print('#Gen block4:', block4.size()) 
        block5 = self.block5(block4)
        # print('#Gen block5:', block5.size()) 
        block6 = self.block6(block5)
        # print('#Gen block6:', block6.size()) 
        block7 = self.block7(block6)
        # print('#Gen block7:', block7.size())
        block8 = self.block8(block1 + block7)
        # print('#Gen block8:', block8.size())
        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class JpegComp(nn.Module):
    def __init__(self):
        super(JpegComp, self).__init__()

    def forward(self, input_, qf):
        return jpegLayer(input_, qf)
