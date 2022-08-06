from train import *
para = dict()
para['seed'] = 2022
para['average_times'] = 3
para['num_layers'] = 2
para['epochs'] = 3200
para['batch_size'] = 1
para['lr'] = 1e-4
para['kernel_size'] = 2
para['data_path'] = "DRIVE/"
para['adqc'] = {'down1': [0, 0], 'down2': [0, 0],
                'down3': [0, 0], 'down4': [0, 0],
                'up1': [0, 0], 'up2': [0, 0],
                'up3': [0, 0], 'up4': [0, 0]}
para['model_name'] = 'aa'
para['checkpoint'] = False
if __name__ == '__main__':
    manytimes(para)
