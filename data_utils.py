from os import listdir
from os.path import join

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, transforms
from utils.jpeg_layer import jpeg_compression_transform, simg_jpeg_compression, jpeg_compression
# from utils.custom_trasform import NRandomCrop

from numpy import asarray, clip

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def val_hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

# def train_multiple_hr_transform(crop_size, crop_numb, padding=0):
#     return Compose([
#         NRandomCrop(size=crop_size, n=crop_numb, padding=padding),
#         transforms.Lambda(
#             lambda crops: torch.stack([
#                 transforms.ToTensor()(crop)
#                 for crop in crops
#             ]))
#     ])

def jr_transform(quality_factor):
    return Compose([
        jpeg_compression_transform(quality_factor)
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        # Resize(400),
        # CenterCrop(400),
        ToTensor()
    ])


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.xavier_uniform_(m.bias)
        else:
            m.bias.data.zero_()

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, quality_factor, train=True, crop_numb=1, padding=0):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)] * crop_numb
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        # self.hr_transform = train_multiple_hr_transform(crop_size, crop_numb, padding)
        self.hr_transform = train_hr_transform(crop_size) if train else val_hr_transform(crop_size)
        self.quality_factor = quality_factor
        # self.jr_transform = jr_transform(quality_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        jr_image = simg_jpeg_compression(hr_image, self.quality_factor)
        return jr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, quality_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.quality_factor = quality_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.crop_size = crop_size
        # self.jr_transform = jr_transform(quality_factor)

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        # crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)

        hr_image = ToTensor()(CenterCrop(self.crop_size)(hr_image))
        jr_image = simg_jpeg_compression(hr_image, self.quality_factor)

        return jr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

def scalePixels(image):
    pixels = asarray(image.cpu())
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate global mean and standard deviation
    mean, std = pixels.mean(), pixels.std()
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    # clip pixel values to [-1,1]
    pixels = clip(pixels, -1.0, 1.0)
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    return torch.Tensor(pixels).cuda()