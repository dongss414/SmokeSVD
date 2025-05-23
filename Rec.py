from resconstruction import test_reconstruction
import torch
import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed)  # Python 原生随机数
    np.random.seed(seed)  # numpy 随机数
    torch.manual_seed(seed)  # CPU 上的 torch 随机数
    torch.cuda.manual_seed(seed)  # GPU 上的 torch 随机数
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True  # 保证卷积等操作可复现
    torch.backends.cudnn.benchmark = False  # 禁用自动算法优化（也有助于可复现）


def main(model_config = None):
    modelConfig = {
        "state": "reconstruction", # or train_vel or reconstruction or train_vort or train_den
        "epoch": 300,
        "batch_size": 1,
        "lr": 5e-5,
        "grad_clip": 1.,
        'dropout': 0,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        'useGrho':True,
        'usePredict':True,
        'img_size' : (112,64),
        'den_size' : (64,112,64),
        'vel_size': (3, 64,112,64),
        'cube_size': 64,
        'scalarflow_folder_prefix':"sim_",
        'scalarflow_index':[11,12,13,14,15,16,17, 18],
        'scalarflow_valindex':[19],
        'dataset':"scalarflow",#'scalarflow',
        'trainscalar':False,
        'start_frame':20,
        'end_frame': 139,
        'scene_num':[89],
        'denpath':'/home/dongss/pycharmProjects/fluidResconstruction/randomSourceData/density',
        'use_img_num':16,
        }
    if model_config is not None:
        modelConfig = model_config

    if modelConfig["state"] == "reconstruction":
        if modelConfig['dataset'] == 'syn':
            for i in modelConfig["scene_num"]:
                test_reconstruction(modelConfig,scene_num=i)
        else:
            for i in modelConfig["scalarflow_valindex"]:
                test_reconstruction(modelConfig,scene_num=i)





if __name__ == '__main__':
    set_seed(42)
    main()
