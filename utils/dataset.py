import random

import numpy as np
import torch
import os
import glob

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2


class ISBI_Loader(Dataset):
    def __init__(self, root_path, mode='train'):
        # 初始化函数，读取所有data_path下的图片
        self.transforms = None
        if mode == 'train':
            sub_dir = 'training'
            self.transforms = (
                # RandomCrop((256, 256)),  # transforms.compose(_),用来串联连续多个图像处理操作，类似于nn.Sequential
                flip(),
                Rotate()
            )
            # RandomRotate())
        elif mode == 'test':
            sub_dir = 'test'
        else:
            sub_dir = 'val'
        self.data_path = root_path
        self.imgs_path = glob.glob(os.path.join(root_path, sub_dir, 'images', '*.tif'))
        self.manual_path = os.path.join(root_path, sub_dir, '1st_manual/')
        self.mask_path = os.path.join(root_path, sub_dir, 'mask/')

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        manual_path = self.manual_path + os.path.basename(image_path)[:3] + 'manual1.gif'
        # mask_path = self.mask_path + os.path.basename(image_path)[:3] + 'training_mask.gif'
        manual = Image.open(manual_path)
        manual = np.array(manual) / 255
        # roi_mask = Image.open(mask_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255
        '''plt.figure("figure name screenshot")  # 图像窗口名称
        plt.imshow(image)
        plt.show()'''
        if self.transforms:
            for t in self.transforms:
                image, manual = t(image, manual)
        image = image.reshape(1, image.shape[0], image.shape[1])
        manual = manual.reshape(1, manual.shape[0], manual.shape[1])
        return image, manual

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


class RandomCrop:
    def __init__(self, shape):
        self.shape = [shape, shape] if isinstance(shape, int) else shape
        self.fill = 0
        self.padding_mode = 'constant'

    def _get_range(self, shape, crop_shape):
        if shape == crop_shape:
            start = 0
        else:
            start = random.randint(0, shape - crop_shape)
        end = start + crop_shape
        return start, end

    def __call__(self, img, mask):

        _, h, w = img.shape
        sh, eh = self._get_range(h, self.shape[0])
        sw, ew = self._get_range(w, self.shape[1])
        return img[:, sh:eh, sw:ew], mask[:, sh:eh, sw:ew]


class flip:
    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip

    def __call__(self, image, label):
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label


class Rotate:
    def __init__(self, limit=None):
        self.limit = limit
        if limit is None:
            self.limit = [-90, 90]

    def _rotate(self, img, angle):
        height, width = img.shape[0:2]
        mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, mat, (height, width),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        return img

    def __call__(self, image, mask):
        angle = random.uniform(self.limit[0], self.limit[1])
        return self._rotate(image, angle), self._rotate(mask, angle)


if __name__ == '__main__':
    isbi_dataset = ISBI_Loader('DRIVE/')
    print(len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=True)

    for image, mask in train_loader:
        break
