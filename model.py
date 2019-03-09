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
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(512, n_out, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.additional_layers(out)
        return out
