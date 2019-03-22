import torch
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


class OCRResNet34(nn.Module):
    def __init__(self, n_out, pretrained=True):
        super(OCRResNet34, self).__init__()
        resnet34 = models.resnet34(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet34.children())[:-2])

        self.additional_layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
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


class OCRResNet50(nn.Module):
    def __init__(self, n_out, pretrained=True):
        super(OCRResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])

        self.additional_layers = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
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


class OCRVGG19(nn.Module):
    def __init__(self, n_out, pretrained=True):
        super(OCRVGG19, self).__init__()
        vgg19 = models.vgg19_bn(pretrained=pretrained)
        self.feature_extractor = list(vgg19.children())[0][:-1]

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


class OCRVGG16(nn.Module):
    def __init__(self, n_out, pretrained=True):
        super(OCRVGG16, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=pretrained)
        self.feature_extractor = list(vgg16.children())[0][:-1]

        self.additional_layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
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
        for layer in self.downsampler:
            x = layer(x)
        return x


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
        for layer in self.upsampler:
            x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_in):
        super(UNet, self).__init__()

        self.encoder1 = DownSampler(n_in, n_in)
        self.encoder2 = DownSampler(n_in, n_in)
        self.encoder3 = DownSampler(n_in, n_in)
        self.encoder4 = DownSampler(n_in, n_in)

        self.decoder1 = UpSampler(n_in, n_in, use_bn=False)
        self.decoder2 = UpSampler(2*n_in, n_in)
        self.decoder3 = UpSampler(2*n_in, n_in)
        self.decoder4 = UpSampler(2*n_in, n_in)

    def forward(self, x):
        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(out_encoder1)
        out_encoder3 = self.encoder3(out_encoder2)
        out_encoder4 = self.encoder4(out_encoder3)

        out_decoder1 = self.decoder1(out_encoder4)
        out_decoder2 = self.decoder2(torch.cat([out_decoder1, out_encoder3], dim=1))
        out_decoder3 = self.decoder3(torch.cat([out_decoder2, out_encoder2], dim=1))
        out_decoder4 = self.decoder4(torch.cat([out_decoder3, out_encoder1], dim=1))

        return out_decoder4


class OCRUNet18(nn.Module):
    def __init__(self, n_out, pretrained=True):
        super(OCRUNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-2])

        self.unet = UNet(512)

        self.additional_layers = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(1024, n_out, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        resnet_out = self.feature_extractor(x)
        unet_out = self.unet(resnet_out)
        out = self.additional_layers(torch.cat([resnet_out, unet_out], dim=1))
        return out