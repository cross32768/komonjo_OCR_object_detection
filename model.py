from torch import nn
from torchvision import models


class OCRResNet18(nn.Module):
    def __init__(self, n_out, pretrained=True):
        super(OCRResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-2])

        self.additional_layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, n_out, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.additional_layers(out)
        return out


class DownSampler(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=4, stride=2, padding=1, use_bn=True, drop_rate=0):
        super(DownSampler, self).__init__()
        self.downsampler = nn.ModuleList()

        self.downsampler.append(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding))
        if use_bn:
            self.downsampler.append(nn.BatchNorm2d(n_out))
        if drop_rate != 0:
            self.downsampler.append(nn.Dropout2d(drop_rate))
        self.downsampler.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        out = self.downsampler(x)
        return out


class UpSampler(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=4, stride=2, padding=1, use_bn=True, drop_rate=0):
        super(UpSampler, self).__init__()
        self.upsampler = nn.ModuleList()

        self.upsampler.append(nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding))
        if use_bn:
            self.upsampler.append(nn.BatchNorm2d(n_out))
        if drop_rate != 0:
            self.upsampler.append(nn.Dropout2d(drop_rate))
        self.upsampler.append(nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.upsampler(x)
        return out


class UNet(nn.Module):
    def __init__(self, n_in, max_n_hidden, depth):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

    def forward(self, x):
        1+1


class OCRUNet(nn.Module):
    def __init__(self, n_out, image_size):
        super(OCRUNet, self).__init__()

    def forward(self, x):
        1+1
