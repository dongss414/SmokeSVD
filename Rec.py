from resconstruction import test_reconstruction
import torch
import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


def main():
    modelConfig = {
        "state": "reconstruction",
        "batch_size": 1,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        'useGrho':True,
        'usePredict':True,
        'img_size' : (112,64),
        'den_size' : (64,112,64),
        'vel_size': (3, 64,112,64),
        'dataset':"scalarflow",#'scalarflow',
        'start_frame':20,
        'end_frame': 140,
        'scene_num':[0],
        'use_img_num':16,
        }

    if modelConfig["state"] == "reconstruction":
        for i in modelConfig["scene_num"]:
            test_reconstruction(modelConfig,scene_num=i)





if __name__ == '__main__':
    set_seed(666)
    main()
