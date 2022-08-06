""" Parts of the U-Net model_adqc """
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from model_adqc.adqc import adqc_conv2d


# warnings.filterwarnings("ignore", category=UserWarning)
class SpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SpConv2d, self).__init__()
        self.padding = int((kernel_size-2)/2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding)

    def forward(self, x):
        n, c, h, w = x.size()
        assert c % 4 == 0
        x1 = x[:, :c // 4, :, :]
        x2 = x[:, c // 4:c // 2, :, :]
        x3 = x[:, c // 2:c // 4 * 3, :, :]
        x4 = x[:, c // 4 * 3:c, :, :]
        x1 = nn.functional.pad(x1, (1, 0, 1, 0), mode="constant", value=0)  # left top
        x2 = nn.functional.pad(x2, (0, 1, 1, 0), mode="constant", value=0)  # right top
        x3 = nn.functional.pad(x3, (1, 0, 0, 1), mode="constant", value=0)  # left bottom
        x4 = nn.functional.pad(x4, (0, 1, 0, 1), mode="constant", value=0)  # right bottom
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)


def Conv2d(flag, in_channels, out_channels, layers=3, kernel=2):
    if flag == 0:
        conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        conv = nn.Sequential(
            adqc_conv2d(in_channels, out_channels, layers=layers, kernerl_size=kernel),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    return conv

class doubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),stride=(1,1),padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),stride=(1,1),padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, adqc, layers=3, kernel=4):
        super().__init__()
        self.layers = layers
        self.in_conv = Conv2d(adqc[0], in_channels, out_channels, layers, kernel)
        self.out_conv = Conv2d(adqc[1], out_channels, out_channels, layers, kernel)

    def forward(self, x):
        x = self.in_conv(x)
        return self.out_conv(x)


class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, adqc, layers=3, kernel=4):
        super().__init__()
        self.layers = layers
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, adqc=adqc, layers=self.layers, kernel=kernel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, adqc, layers=3, kernel=4):
        super().__init__()
        self.layers = layers
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, adqc, layers=3, kernel=4):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, adqc=adqc, layers=layers, kernel=kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = int(torch.tensor([x2.size()[2] - x1.size()[2]]))
        diffX = int(torch.tensor([x2.size()[3] - x1.size()[3]]))
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, adqc, layers=3, kernel=4):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = int(torch.tensor([x2.size()[2] - x1.size()[2]]))
        diffX = int(torch.tensor([x2.size()[3] - x1.size()[3]]))
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)
