from torch.autograd import Function
import random
from io import BytesIO
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch

class jpeg_compression_transform(object):

    def __init__(self, quality_factor):
        self.quality_factor = quality_factor
        
    def __call__(self, img):
        output = simg_jpeg_compression(img, self.quality_factor)
        return output

    def __repr__(self):
        return self.__class__.__name__+'()'


def simg_jpeg_compression(image, qf):
    imgTensor = torch.zeros_like(image)
    image = TF.to_pil_image(image.cpu())
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    image_comp = Image.open(outputIoStream)
    imgTensor[: , :, :] = TF.to_tensor(image_comp)
    return imgTensor


def jpeg_compression(images, qf):
    imgsTensor = torch.zeros_like(images)
    for i, image in enumerate(images):
        image = TF.to_pil_image(image.cpu())
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        image_comp = Image.open(outputIoStream)
        imgsTensor[i, :, :, :] = TF.to_tensor(image_comp)
    return imgsTensor.cuda()


class JpegLayerFun(Function):
    @staticmethod
    def forward(ctx, input_, qf):
        ctx.save_for_backward(input_)
        output = jpeg_compression(input_, qf)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input, None


jpegLayer = JpegLayerFun.apply
