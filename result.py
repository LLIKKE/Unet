from eval import *


def result(adqc_model, para):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    para_ = para.copy()
    net = UNet(n_channels=1, n_classes=1, layers=para_['num_layers'], adqc=para_['adqc'],
               kernel_size=para_['kernel_size'])
    net.to(device=device)
    net.load_state_dict(torch.load(adqc_model, map_location=device))
    net.eval()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter_adqc: %.5fM" % (total / 1e6))
    print("-----------按照train_loss保存")
    _,adqc_test = predict_sum(model=net, data='DRIVE/', para=para_,mode='test')
    print('adqc_test', adqc_test)
    net.load_state_dict(torch.load(f'vail_{adqc_model}', map_location=device))
    _,vail_adqc_test = predict_sum(model=net, data='DRIVE/', para=para_,mode='test')
    print("-----------按照vail_loss保存")
    print('adqc_test', vail_adqc_test)

    return adqc_test, vail_adqc_test


if __name__ == '__main__':
    para = dict()
    para['seed'] = 2032
    para['num_layers'] = 1
    para['epochs'] = 35
    para['batch_size'] = 1
    para['lr'] = 0.0001
    para['kernel_size'] = 2
    para['data_path'] = "dataset/data/"
    para['average_times'] = 1
    para['adqc'] = {'down1': [0, 0], 'down2': [0, 0],
                    'down3': [0, 0], 'down4': [1, 1],
                    'up1': [1, 0], 'up2': [0, 0],
                    'up3': [0, 0], 'up4': [0, 0]}
    para['model_name'] = 'unet'
    comput_loss = dict()
    train_model_data = []
    train_model_test = []
    vail_model_data = []
    vail_model_test = []
    p = 0
    for i in range(para['average_times']):
        train, vail = result(adqc_model=f"{para['model_name']}_{i + 1}.pth", para=para)
        train_model_data.append(train[0])  # train[0]里面数组4个指标
        train_model_test.append(train[1])
        vail_model_data.append(vail[0])
        vail_model_test.append(vail[1])
    comput_loss['train_model_data'] = np.array(train_model_data)
    comput_loss['train_model_test'] = np.array(train_model_test)
    comput_loss['vail_model_data'] = np.array(vail_model_data)
    comput_loss['vail_model_test'] = np.array(vail_model_test)
    assess_ = ['loss', 'PA', 'IOU', 'Dice']
    for key in comput_loss:
        mean = np.mean(comput_loss[key], axis=0)
        std = np.std(comput_loss[key], axis=0)
        print('in_the_end')
        for p in range(len(assess_)):
            print(f'{key},[{assess_[p]}] mean:{mean[p]},std:{std[p]}')
