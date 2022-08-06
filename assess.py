'''像素准确率(Pixel Accuracy, PA)、
交并比(Intersection-Over-Union，IOU)、
Dice系数(Dice Coeffcient),
豪斯多夫距离（ Hausdorff distance，HD95），
体积相关误差（relative volume error, RVE）'''
import numpy as np
from matplotlib import pyplot as plt


def binary_pa(s, g):
    s = np.array(s.data.cpu()[0])[0]
    g = np.array(g.data.cpu()[0])[0]

    """
        calculate the pixel accuracy of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
    pa = ((s == g).sum()) / g.size
    return pa


# IOU evaluation 交并比
def binary_iou(s, g):
    s = np.array(s.data.cpu()[0])[0]
    g = np.array(g.data.cpu()[0])[0]
    # assert (len(s.shape), len(g.shape))
    # 两者相乘值为1的部分为交集
    intersecion = np.multiply(s, g)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(s + g > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou


# 骰子系数、dice、f1-score
def binary_dice(s, g):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    s = np.array(s.data.cpu()[0])[0]
    g = np.array(g.data.cpu()[0])[0]
    # assert (len(s.shape) == len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
    return dice



