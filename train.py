from common import *
from model_adqc.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch as tc
from eval import predict_sum
from result import result
import time


def train_net(para):
    setup_seed(para['seed'])
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1, layers=para['num_layers'], adqc=para['adqc'],
               kernel_size=para['kernel_size']).to(device=device)
    net.load_state_dict(torch.load('unet.pth', map_location=device))
    print(net)
    print(torch.load('unet.pth', map_location=device))
    count_parameters(net)
    isbi_dataset = ISBI_Loader(para['data_path'])
    train_loader = tc.utils.data.DataLoader(dataset=isbi_dataset,
                                            batch_size=para['batch_size'],
                                            shuffle=True)
    optimizer = optim.RMSprop(net.parameters(), lr=para['lr'], weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    stopping = EarlyStopping(patience=3, delta=2, way='normal')
    best_loss = float('inf')
    iters = 0
    start_ = time.time()
    last_epoch, train_loss, v_loss = resume(checkpoint=para['checkpoint'], net=net, optimizer=optimizer)
    for epoch in range(last_epoch, para['epochs']):
        for p, data in enumerate(train_loader):
            warm_lr(optimizer, para['lr'], gamma=0.5, epoch=epoch, i=p, dataloader=train_loader,
                    milestones=[100, 500, 1000, 3000, 5000, 10000])
            optimizer.zero_grad()
            image, label = data[0].to(device=device, dtype=tc.float32), data[1].to(device=device, dtype=tc.float32)
            pred = net(image)
            loss = criterion(pred, label)
            if iters % 10 == 0:
                train_loss.append(loss.item())
            print(iters, loss.item())
            draw(list(range(0, len(train_loss) * 10, 10)), train_loss, named=f'{i+1}_train', y_limit=[0, 0.7])
            if loss < best_loss:
                best_loss = loss
                tc.save(net.state_dict(), f"{para['model_name']}_{i + 1}.pth")
            loss.backward()
            optimizer.step()
            iters += 1
        if epoch < 600 and epoch % 100 == 0:
            now_loss, _ = predict_sum(model=net, data='DRIVE/', para=para, mode='eval')
            v_loss.append(now_loss)
        elif epoch >= 600 and epoch % 50 == 0:
            now_loss, _ = predict_sum(model=net, data='DRIVE/', para=para, mode='eval')
            v_loss.append(now_loss)
            stopping(now_loss, net, f"{para['model_name']}_{i + 1}.pth", v_loss)
        if epoch > 800 and epoch % 100 == 0:
            save_checkpoint(net, optimizer, epoch, train_loss, v_loss)
        if stopping.early_stop:
            break
    end_ = time.time()

    print('-----验证集损失-----')
    val_iter = list(range(0, 600, 100)) + list(range(600, 600 + (len(v_loss) - 6) * 50, 50))
    for t in range(len(val_iter)):
        print(val_iter[t], v_loss[t])
    draw(val_iter, v_loss, named=f'{i + 1}_vail', y_limit=[0, 0.7])
    print(f'第{i + 1}次训练花费:', (end_ - start_) / 60 / 60, 'h')
    print('train_end')


def manytimes(para=None):
    comput_loss = dict()
    train_model_test = []
    vail_model_test = []
    para_ = para.copy()

    global i
    for i in range(para_['average_times']):
        para_['seed'] = 2022 + i * 10
        # net.load_state_dict(torch.load('best_model.pth', map_location=device))
        train_net(para=para_)
        train, vail = result(adqc_model=f"{para_['model_name']}_{i + 1}.pth", para=para_)
        train_model_test.append(train)
        vail_model_test.append(vail)
    comput_loss['train_model_test'] = train_model_test
    comput_loss['vail_model_test'] = vail_model_test
    assess_ = ['loss', 'PA', 'IOU', 'Dice']
    for key in comput_loss:
        mean = np.mean(comput_loss[key], axis=0)
        std = np.std(comput_loss[key], axis=0)
        for p in range(len(assess_)):
            print(f'{key},[{assess_[p]}] mean:{mean[p]},std:{std[p]}')
