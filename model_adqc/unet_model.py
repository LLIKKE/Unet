from model_adqc.unet_parts import *
import torch as tc


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, layers, adqc, kernel_size=2):
        super(UNet, self).__init__()
        self.adqc = adqc.copy()
        self.inc = doubleConv(n_channels, 64)
        self.down1 = Down(64, 128, layers=layers, adqc=self.adqc['down1'], kernel=kernel_size)
        self.down2 = Down(128, 256, layers=layers, adqc=self.adqc['down2'], kernel=kernel_size)
        self.down3 = Down(256, 512, layers=layers, adqc=self.adqc['down3'], kernel=kernel_size)
        self.down4 = Down(512, 512, layers=layers, adqc=self.adqc['down4'], kernel=kernel_size)
        self.up1 = Up(1024, 256, layers=layers, adqc=self.adqc['up1'], kernel=kernel_size)
        self.up2 = Up(512, 128, layers=layers, adqc=self.adqc['up2'], kernel=kernel_size)
        self.up3 = Up(256, 64, layers=layers, adqc=self.adqc['up3'], kernel=kernel_size)
        self.up4 = Up(128, 64, layers=layers, adqc=self.adqc['up4'], kernel=kernel_size)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    device = tc.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    para = dict()
    para['num_layers'] = 1
    para['adqc'] = {'down1': [1, 1], 'down2': [0, 0],
                    'down3': [0, 0], 'down4': [1, 0],
                    'up1': [0, 0], 'up2': [0, 0],
                    'up3': [0, 0], 'up4': [0, 0]}
    net = UNet(n_channels=1, n_classes=1, layers=para['num_layers'], adqc=para['adqc'])
    print(net)
