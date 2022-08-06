import numpy as np
import os
import torch, random
import torch.nn as nn
from PIL import Image
import torch as tc
from matplotlib import pyplot as plt


def draw(xlist, ylist, save_path='', named='train_loss', y_limit=None, title='train_loss_decline', x_tag='iter',
         y_tag='loss', ):
    plt.scatter(xlist, ylist, c='blue')
    plt.title(title)
    if y_limit is not None:
        plt.ylim(y_limit[0], y_limit[1])
    plt.xlabel(x_tag)
    plt.ylabel(y_tag)
    plt.savefig(save_path + f'{named}.png')
    plt.cla()


def count_parameters(model):
    num = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (num / 1e6))
    return num


# Seed for repeatability
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def save_checkpoint(net, optimizer, epoch, losslist=None, val_losslist=None):
    '''save net，optimizer，epoch, loss'''
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
        print('自动创建checkpoint子文件夹')
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': losslist,
             "val_loss": val_losslist}
    if epoch % 200 == 0:
        torch.save(state, f'checkpoint/check_point_{epoch}.pth')


def resume(checkpoint=False, net=None, optimizer=None):
    loss = []
    val_loss = []
    if checkpoint:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('checkpoint/check_point.pth')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        val_loss = checkpoint['val_loss']
        return last_epoch, loss, val_loss
    else:
        last_epoch = 0
        return last_epoch, loss, val_loss


def warm_lr(optimizer, lr, gamma, epoch, i, dataloader, milestones=None):
    # Learning rate
    epoch = epoch + 1
    if milestones is None:
        milestones = [2, 5, 10, 50, 100]
    if epoch <= milestones[0]:
        lr = lr / (len(dataloader) * milestones[0]) * (i + len(dataloader) * epoch)
    elif milestones[0] <= epoch <= milestones[1]:
        lr = lr * gamma
    elif milestones[1] < epoch <= milestones[2]:
        lr = lr * gamma ** 2
    elif milestones[2] < epoch <= milestones[3]:
        lr = lr * gamma ** 3
    elif milestones[3] < epoch <= milestones[4]:
        lr = lr * gamma ** 4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def exp_mov_avg(nets, net, alpha=0.999, global_step=999):
    """ 参数优化：指数平均数 ema_para = ema_para * alpha + param * (1 - alpha)
    nets: 做了ema的参数网络
    net: 训练中实际使用的网络
    global_step: 迭代次数（训练次数）
    使用： 网络实例化时拷贝一个nets = copy.deepcopy(net) ， 然后训练中调用exp_mov_avg(nets, net, global_step=step)
   y.add_(x,alpha) 指定x按比例alpha缩放后，加到y上，并取代y的内存y.mul_(x,other) 将x乘以other，取代y的内存存储，这里用带’_‘的函数就已经取代原参数了"""
    '''使用的时候，需额外定义一个nets'''
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(nets.parameters(), net.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, way="normal"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            验证集连续多少次loss上升会停止训练
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            采用GL标准，这个delta意味着刚计算的验证集loss比最好的验证集loss变差了 delta %
                            Default: 0
            way ():两种停止标准
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.way = way

    def __call__(self, val_loss, model, model_name, val_losslist=None):

        if val_losslist is None:
            val_losslist = []
        score = val_loss
        if len(val_losslist) >= 4:
            if abs(round(val_losslist[-4:-1][0], 4) - round(val_losslist[-4:-1][1], 4)) < 10e-12 and abs(
                    round(val_losslist[-4:-1][1], 4) - round(val_losslist[-4:-1][2], 4)) < 10e-12:
                self.early_stop = True
                print('损失稳定')
        if self.way == 'normal':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, model_name)
            elif score > self.best_score or abs(score - self.best_score) < 1e-12:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    print('损失上升')
            elif score < self.best_score:
                self.best_score = score
                self.counter = 0
                self.save_checkpoint(val_loss, model, model_name)
        if self.way == "PQ_alpha":
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, model_name)
            GL = 100 * (score / self.best_score - 1)
            P_K = 1000 * (sum(val_losslist[-self.patience - 1:-1]) / (
                    self.patience * min(val_losslist[-self.patience - 1:-1]) - 1))
            PQ_alpha = GL / P_K
            if PQ_alpha > self.delta:
                self.save_checkpoint(val_loss, model, model_name)

    def save_checkpoint(self, val_loss, model, model_name):
        """
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        """
        tc.save(model.state_dict(), f'vail_{model_name}')  # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss


# def save_config():


# params initialization
def weight_initV1(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def weight_initV2(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weight_initV3(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


if __name__ == '__main__':
    print(os.path.exists('checkpoint\\'))
