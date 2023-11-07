import torch
import numpy as np
import torchvision.transforms.functional_tensor as F_t
import torchvision.transforms.functional as F

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def set_p(self, p):
        self.p = p

    def forward(self, item):


        if torch.rand(1) > self.p:
            return item
        img, labels = item
        img = F.hflip(img)

        labels[..., [3, 1]] = 1 - labels[..., [1, 3]]

        return (img, labels)


class RandomGasussianBlur(torch.nn.Module):
    def __init__(self, sigma = [0.1,9],kernel_size = 7, p=0.1):
        super().__init__()
        self.p = p

        self.sigma_min = sigma[0]
        self.sigma_max = sigma[1]

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size value should be an odd and positive number.")
        self.kernel_size = kernel_size

    def set_p(self, p):
        self.p = p

    def get_param(self):
        sigma_max = int(torch.randint(1, int(self.sigma_max), (1,)))
        sigma = [self.sigma_min, sigma_max]
        kernel_size = int(torch.randint(3, int(self.kernel_size), (1,)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = [kernel_size, kernel_size]
        return sigma, kernel_size

    def forward(self, items):

        if torch.rand(1) > self.p:
            return items
        img, labels = items
        sigma, kernel_size = self.get_param()
        img = F.gaussian_blur(img, kernel_size, sigma)
        return (img, labels)


class RandomColorJitter(torch.nn.Module):
    def __init__(self, brightness=[0.25,1.3],contrast=[0.2,3.0]
                 ,saturation=[0.0,8.0], sharpness=20,p=0.1):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.sharpness = sharpness
    def set_p(self, p):
        self.p = p

    def get_param(self):
        brightness = float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        contrast = float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
        saturation = float(torch.empty(1).uniform_(self.saturation[0], self.saturation[1]))
        sharpness = float(torch.empty(1).uniform_(0, self.sharpness ))
        return brightness, contrast, saturation,   sharpness

    def forward(self, items):

        if torch.rand(1) > self.p:
            return items
        brightness, contrast, saturation,  sharpness = self.get_param()
        img, labels = items
        pic = img
        index = int(torch.empty(1).uniform_(0, 5))

        if index == 0:
            pic = F_t.adjust_brightness(pic.to(torch.int), brightness)
        if index == 1:
            pic = F_t.adjust_contrast(pic.to(torch.int), contrast)
        if index == 2:
            pic = F_t.adjust_saturation(pic.to(torch.int), saturation)
        if index == 4:
            pic = F_t.adjust_sharpness(pic.to(torch.int), sharpness)
        if index == 5:
            pic = F.rgb_to_grayscale(pic, num_output_channels=3)

        img = pic.to(torch.float32)

        return (img,labels)


class RandomGrayScale(torch.nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def set_p(self, p):
        self.p = p

    def forward(self, items):

        if torch.rand(1) > self.p:
            return items
        img, labels = items
        img = F.rgb_to_grayscale(img, num_output_channels=3)
        return (img, labels)

class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index):
        return self.transforms[index]

    def __call__(self, items):
        for t in self.transforms:
            items = t(items)
        return items

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Transforms:
    def __init__(self):
        self.transforms = Compose(torch.nn.Sequential(
            RandomGasussianBlur(p=0.2),
            RandomHorizontalFlip(p=0.2),
            RandomColorJitter(p=0.5)
            )
        )




    def __call__(self, items):
        items = self.transforms(items)

        return items

