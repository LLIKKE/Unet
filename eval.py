import numpy as np
import torch.nn as nn
import glob
import torch
import cv2
from model_adqc.unet_model import UNet
from assess import *
from utils.dataset import ISBI_Loader


def predict_sum(model, data, para, way='net',mode = 'test'):
    '''
            model   可以之间传入net,这样way='net'，也可以传入模型model.pth的名字
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    para_ = para.copy()
    if way == 'net':
        net = model
    else:
        net = UNet(n_channels=1, n_classes=1, layers=para_['num_layers'], adqc=para_['adqc'])
        net.load_state_dict(torch.load(model, map_location=device))
    net.to(device=device)
    net.eval()
    isbi_dataset = ISBI_Loader(data, mode)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    losssum = 0
    PA = 0
    IOU = 0
    Dice = 0
    a=0
    for image, label in train_loader:
        image, label = image.to(device=device, dtype=torch.float32), label.to(device=device, dtype=torch.float32)
        pred = net(image)
        loss = criterion(pred, label)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        label[label >= 0.3] = 1
        label[label < 0.3] = 0
        PA += binary_pa(pred, label)
        IOU += binary_iou(pred, label)
        Dice += binary_dice(pred, label)
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        a+=1
        cv2.imwrite(f'DRIVE/test/{a}.png', pred)

        losssum += loss.item()
    return losssum / len(isbi_dataset), [losssum / len(isbi_dataset), PA / len(isbi_dataset), IOU / len(isbi_dataset),
                                         Dice / len(isbi_dataset)]
