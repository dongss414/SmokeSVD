import os
import time
import random
import logging
import sys

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
from typing import Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from Data import TestSynthesisDataset,TestScalarflowProcDataset
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from unet3p_model import UNet3D_trainVelocity,UNet3DMultiView16
from advection import *
from predict import predict,InitPredict
from model_reference import UNet,woTUNet
from render import render_scene,render_with_y_axis_rotation_cuda
from fluidResconstruction.HR import hr_test,init_hr
import torchvision.transforms as T



densityPath = "/home/dongss/pycharmProjects/fluidResconstruction/randomSourceData/density"
velocityPath = "/home/dongss/pycharmProjects/fluidResconstruction/randomSourceData/velocity"




def iteration(optimizer,src,den0,den1,vel,t,scheduler,need_print=False,config=0):
    last_src = src
    for i in range(t):
        optimizer.zero_grad()
        nxt_den = advection(den0, vel, src)
        nxt_den = set_zero_border(nxt_den).unsqueeze(0).unsqueeze(0).cuda() # torch.Size([1, 1, 64, 64, 64])

        front = torch.mean(nxt_den, dim=2)
        truefront = torch.mean(den1, dim=2)
        denloss = F.mse_loss(nxt_den, den1)
        frontloss = F.mse_loss(front, truefront)
        srcloss =  denloss + frontloss + F.mse_loss(last_src,src)


        # if i == t-1 and need_print:
        #     print(f'iter: {t}, frontloss: {frontloss}, denloss: {denloss}')
        srcloss.backward()
        torch.nn.utils.clip_grad_norm_([src], max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-5)
    # if need_print:
    #     for param_group in optimizer.param_groups:
    #         print(f'Learning rate: {param_group["lr"]}')

    return src.data


def setup_logger(log_path,name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # 设置日志级别
    # 创建一个文件处理器，将日志写入文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # 创建一个控制台处理器（可选）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个格式化器，并将其添加到处理器中
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger 中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def test_reconstruction(modelConfig: Dict,scene_num=81):
    start_time = time.time()
    use_img_num = modelConfig['use_img_num'] #2,4,8,16
    if modelConfig['dataset'] == 'scalarflow':
        SFflag = True
    else:
        SFflag = False
    angle_idx = []
    angle_idx = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    sr_idx =  [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    prev_prompt_dict = {
        1: torch.zeros(1, 64, 64, dtype=torch.float32),
        2: torch.zeros(1, 64, 64, dtype=torch.float32),
        3: torch.zeros(1, 64, 64, dtype=torch.float32),
        4: torch.zeros(1, 64, 64, dtype=torch.float32),
        5: torch.zeros(1, 64, 64, dtype=torch.float32),
        6: torch.zeros(1, 64, 64, dtype=torch.float32),
        7: torch.zeros(1, 64, 64, dtype=torch.float32),
        9: torch.zeros(1, 64, 64, dtype=torch.float32),
        10: torch.zeros(1, 64, 64, dtype=torch.float32),
        11: torch.zeros(1, 64, 64, dtype=torch.float32),
        12: torch.zeros(1, 64, 64, dtype=torch.float32),
        13: torch.zeros(1, 64, 64, dtype=torch.float32),
        14: torch.zeros(1, 64, 64, dtype=torch.float32),
        15: torch.zeros(1, 64, 64, dtype=torch.float32),
    }
    prompt_dict = {
        1: torch.zeros(1, 64, 64, dtype=torch.float32),
        2: torch.zeros(1, 64, 64, dtype=torch.float32),
        3: torch.zeros(1, 64, 64, dtype=torch.float32),
        4: torch.zeros(1, 64, 64, dtype=torch.float32),
        5: torch.zeros(1, 64, 64, dtype=torch.float32),
        6: torch.zeros(1, 64, 64, dtype=torch.float32),
        7: torch.zeros(1, 64, 64, dtype=torch.float32),
        9: torch.zeros(1, 64, 64, dtype=torch.float32),
        10: torch.zeros(1, 64, 64, dtype=torch.float32),
        11: torch.zeros(1, 64, 64, dtype=torch.float32),
        12: torch.zeros(1, 64, 64, dtype=torch.float32),
        13: torch.zeros(1, 64, 64, dtype=torch.float32),
        14: torch.zeros(1, 64, 64, dtype=torch.float32),
        15: torch.zeros(1, 64, 64, dtype=torch.float32),
    }
    now_sr_dict = {
        1: torch.zeros(1, 64, 64, dtype=torch.float32),
        2: torch.zeros(1, 64, 64, dtype=torch.float32),
        3: torch.zeros(1, 64, 64, dtype=torch.float32),
        4: torch.zeros(1, 64, 64, dtype=torch.float32),
        5: torch.zeros(1, 64, 64, dtype=torch.float32),
        6: torch.zeros(1, 64, 64, dtype=torch.float32),
        7: torch.zeros(1, 64, 64, dtype=torch.float32),
        9: torch.zeros(1, 64, 64, dtype=torch.float32),
        10: torch.zeros(1, 64, 64, dtype=torch.float32),
        11: torch.zeros(1, 64, 64, dtype=torch.float32),
        12: torch.zeros(1, 64, 64, dtype=torch.float32),
        13: torch.zeros(1, 64, 64, dtype=torch.float32),
        14: torch.zeros(1, 64, 64, dtype=torch.float32),
        15: torch.zeros(1, 64, 64, dtype=torch.float32),
    }
    view_dict = {
        4: [1,7,9,15],
        8: [2,6,10,14],
        16: [3,4,5,11,12,13],
        # 16: [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    }

    path = "./log"
    folder_name = f'viewNum{use_img_num:02d}_{time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))}'
    savepath = os.path.join(path, folder_name)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    log_filename = f'aaalogfile.log'
    log_path = os.path.join(savepath, log_filename)
    logger = setup_logger(log_path,str(start_time))
    logger.info(f'############ multiViewNum: {use_img_num} ############')
    logger.info(f'############ scene num: {scene_num} ############')

    device = torch.device(modelConfig["device"])

    if modelConfig['dataset'] == "scalarflow":
        mydata = TestScalarflowProcDataset(modelConfig, path=f'/home/dongss/pycharmProjects/Global-Flow-Transport-main/data/ScalarFlow/sim_{scene_num:06d}/input/postprocessed',
                                      imagetransform=transforms.Compose([
                                          transforms.Resize((64,64)),

                                      ]),start_frame=modelConfig['start_frame'], end_frame=modelConfig['end_frame'] )

    else:
        mydata = TestSynthesisDataset(modelConfig,densityPath,velocityPath,
                                        imagetransform=transforms.Compose([
                                            transforms.Resize((64,64)),
                                     ]),start_frame=modelConfig['start_frame'], end_frame=modelConfig['end_frame'] ,scene_num=scene_num)

    dataloader = DataLoader(mydata,batch_size=1,shuffle=False)
    velModel = UNet3D_trainVelocity(inchannels=2).to(device)
    if modelConfig['dataset'] == "scalarflow":
        # velModel.load_state_dict(
        #     torch.load(os.path.join(
        #         "/home/dongss/pycharmProjects/fluidResconstruction/CheckpointsVelocity/SF_[96,176,96]/20250324_205408",
        #         "velocity_wo_multiview_10_.pt"),
        #         map_location=device))
        velModel.load_state_dict(
            torch.load(os.path.join(
                "/home/dongss/pycharmProjects/fluidResconstruction/CheckpointsVelocity/mulFrame",
                "velocity_test_45_.pt"),
                map_location=device))


    else:

        velModel.load_state_dict(
            torch.load(os.path.join(
                "/home/dongss/pycharmProjects/fluidResconstruction/CheckpointsVelocity/mulFrame",
                "velocity_test_45_.pt"),
                map_location=device))

    denModel = UNet3DMultiView16(fe=[32, 64, 128, 256, 512],modelconfig=modelConfig).to(device)
    if modelConfig['dataset'] == "scalarflow":
        denModel.load_state_dict(
            torch.load(os.path.join(
                "/home/dongss/pycharmProjects/fluidResconstruction/CheckpointsMultiViewDensity/SF20250225_165238",
                "SF_noConvT_multiView16_[32, 64, 128, 256, 512]_5_.pt"),
                map_location=device))
    else:
        denModel.load_state_dict(
            torch.load(os.path.join(
                "/home/dongss/pycharmProjects/fluidResconstruction/CheckpointsMultiViewDensity/20250212_203739",
                "Syn_noConvT_multiView16_[32, 64, 128, 256, 512]_11_.pt"),
                       map_location=device))

    denMulModel = UNet3DMultiView16(fe=[32, 64, 128, 256, 512],modelconfig=modelConfig).to(device)
    if modelConfig['dataset'] == "scalarflow":
        denMulModel.load_state_dict(
            torch.load(os.path.join(
                "/home/dongss/pycharmProjects/fluidResconstruction/CheckpointsMultiViewDensity/SF20250225_165238",
                "SF_noConvT_multiView16_[32, 64, 128, 256, 512]_29_.pt"),
                       map_location=device))
    else:
        denMulModel.load_state_dict(
            torch.load(os.path.join("/home/dongss/pycharmProjects/fluidResconstruction/CheckpointsMultiViewDensity/20250222_152438",
                                    "Syn_noConvT_multiView16_[32, 64, 128, 256, 512]_70_.pt"),
                       map_location=device))
    predictModel = UNet(T=1000, ch=32, ch_mult=[1, 2, 3, 4],
                 attn=[2],
                 num_res_blocks=2, dropout=0.,inchannels=2,outchannels=1).to(device)
    if modelConfig['dataset'] == "scalarflow":
        ckpt = torch.load(os.path.join(
            "/home/dongss/pycharmProjects/DenoisingDiffusionProbabilityModel_ddpm__main/CheckpointsForScalarflowDDPM/img5_20250509_140021",
            "1_ch32_multiSource_94_.pt"), map_location=device)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    else:
        ckpt = torch.load(os.path.join(
            "/home/dongss/pycharmProjects/DenoisingDiffusionProbabilityModel_ddpm__main/CheckpointsForSynDensityDDPM/img5_20250514_140708_syn",
            "1_ch32_multiSource_50_all.pt"), map_location=device)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    predictModel.load_state_dict(ckpt)

    predictInitModel = UNet(T=1000, ch=32, ch_mult=[1, 2, 3, 4],
                        attn=[2],
                        num_res_blocks=2, dropout=0., inchannels=2, outchannels=1).to(device)
    if modelConfig['dataset'] == "scalarflow":
        ckpt = torch.load(os.path.join(
            "/home/dongss/pycharmProjects/DenoisingDiffusionProbabilityModel_ddpm__main/CheckpointsForScalarflowDDPM",
            "initrainSF_20250330_144118/init_SFimgProc_79_.pt"), map_location=device)

    else:
        ckpt = torch.load(os.path.join(
            "/home/dongss/pycharmProjects/DenoisingDiffusionProbabilityModel_ddpm__main/CheckpointsForSynDensityDDPM/initrainSyn_20250419_101136",
            "init_Syn_144_.pt"), map_location=device)

    predictInitModel.load_state_dict(ckpt)

    hrModel = woTUNet(ch=32, ch_mult=[1, 2, 3, 4],
                 attn=[2],
                 num_res_blocks=2, dropout=0.,inchannels=1,outchannels=1).to(device)
    if modelConfig['dataset'] == "scalarflow":
        ckpt = torch.load(os.path.join("/home/dongss/pycharmProjects/DenoisingDiffusionProbabilityModel_ddpm__main/CheckpointsForHR/sf_proc_wodiff_20250424_103550",
                    "sf_wodiff_49_.pt"), map_location=device)
        ckpt = torch.load(os.path.join(
            "/home/dongss/pycharmProjects/DenoisingDiffusionProbabilityModel_ddpm__main/CheckpointsForHR/sf_proc_wodiff_20250520_235417",
            "sf_wodiff_12_.pt"), map_location=device)
    else:
        ckpt = torch.load(os.path.join(
            "/home/dongss/pycharmProjects/DenoisingDiffusionProbabilityModel_ddpm__main/CheckpointsForHR/sf_proc_wodiff_20250424_103550",
            "sf_wodiff_49_.pt"), map_location=device)
    hrModel.load_state_dict(ckpt)

    hrInitModel = UNet(T=1000, ch=32, ch_mult=[1, 2, 3, 4],
                 attn=[2],
                 num_res_blocks=2, dropout=0.,inchannels=2,outchannels=1).to(device)
    ckpt = torch.load('/home/dongss/pycharmProjects/DenoisingDiffusionProbabilityModel_ddpm__main/'
                      'CheckpointsForHR/dm_smooth_initsr_20250414_153738/sf_diff_51_.pt', map_location=device)
    hrInitModel.load_state_dict(ckpt)


    hrModel.eval()
    hrInitModel.eval()
    predictModel.eval()
    predictInitModel.eval()
    velModel.eval()
    denModel.eval()
    denMulModel.eval()


    light_positions = [(64, 64, -64)]
    intensity_p = 0
    intensity_a = 1

    def dealrender(density, needrot = False, truescale=0.02, sampescale=0.2, isSample=True,f=64):
        if modelConfig['dataset'] == "scalarflow":
            truescale = 0.01
        else:
            truescale = 0.2

        if isSample:
            front = render_scene(density * sampescale, light_positions, intensity_p, intensity_a, n=0, f=f,
                                 is_side=False)
            side = render_with_y_axis_rotation_cuda(density * sampescale, light_positions, intensity_p, intensity_a, n=0, f=f,
                                             rotation_angle=180)
        else:
            front = render_scene(density * truescale, light_positions, intensity_p, intensity_a, n=0, f=f,
                                 is_side=False)
            side = render_with_y_axis_rotation_cuda(density * truescale, light_positions, intensity_p, intensity_a,
                                                    n=0, f=f,
                                                    rotation_angle=180)
        if needrot:
            front = front.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            front = torch.rot90(front, k=2, dims=(3, 4))#[1,1,1,64,64]
            side = side.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            side = torch.rot90(side, k=2, dims=(3, 4))
        else:
            front = front.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            side = side.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return front, side

    def render(density, sampescale=0.2, isSample=True, isFirst = True, angle = 135.0,f=64,SF_flag = False):
        if modelConfig['dataset'] == "scalarflow":
            truescale = 0.01
        else:
            truescale = 0.2

        if SF_flag:
            angleList = [angle + 30,angle - 30,angle]
        else:
            angleList = [angle + 45, angle - 45, angle]
        if isSample:
            x = render_with_y_axis_rotation_cuda(density * sampescale, light_positions, intensity_p, intensity_a, n=0, f=f,rotation_angle=angleList[0])
            y = render_with_y_axis_rotation_cuda(density * sampescale, light_positions, intensity_p, intensity_a, n=0, f=f,rotation_angle=angleList[1])
            z = render_with_y_axis_rotation_cuda(density * sampescale, light_positions, intensity_p, intensity_a, n=0, f=f,rotation_angle=angleList[2])
        else:
            x = render_with_y_axis_rotation_cuda(density * truescale, light_positions, intensity_p, intensity_a, n=0,
                                                  f=f, rotation_angle=angleList[0])
            y = render_with_y_axis_rotation_cuda(density * truescale, light_positions, intensity_p, intensity_a, n=0,
                                                  f=f, rotation_angle=angleList[1])
            z = render_with_y_axis_rotation_cuda(density * truescale, light_positions, intensity_p, intensity_a, n=0,
                                                  f=f, rotation_angle=angleList[2])


        x = torch.rot90(x.unsqueeze(0).unsqueeze(0), k=2, dims=(2, 3))
        y = torch.rot90(y.unsqueeze(0).unsqueeze(0), k=2, dims=(2, 3))
        z = torch.rot90(z.unsqueeze(0).unsqueeze(0), k=2, dims=(2, 3))

        return x,y,z

    def renderEightViews(density, sampescale=0.2,f=64):

        angleList = [90,115,140,165,190,215,240,265]
        eightImages = []
        for angle in angleList:
            image = render_with_y_axis_rotation_cuda(density * sampescale, light_positions, intensity_p, intensity_a, n=0, f=f,rotation_angle=angle)
            image = torch.rot90(image.unsqueeze(0), k=2, dims=(1, 2))
            eightImages.append(image)
        return eightImages

    def dealsource(dataloader,source,is_optsource=False,prev_prompt_dict=prev_prompt_dict,prompt_dict=prompt_dict,now_sr_dict=now_sr_dict,view_dict=view_dict):
        with tqdm(dataloader, disable=True) as tqdmDataLoader:
            isFirst = True
            flag = False
            denList = []
            epoch_den_loss = 0.0
            epoch_vel_loss = 0.0
            epoch_advected_den_loss = 0.0
            count = 0
            opt = torch.optim.AdamW([source], lr=1e-1)
            scheduler = StepLR(opt, step_size=50, gamma=0.1)
            front_for_grho_List = []
            frontList = []
            trueSideList = []

            for batch in tqdmDataLoader:
                count = count + 1
                num = batch['num']
                trueDen = batch['den'].to(device)
                trueVel = batch['vel'].to(device)

                if modelConfig['dataset'] == "scalarflow":
                    #proc
                    front = batch['front_view'].to(device)
                    side = batch['side_view'].to(device)
                    front = front.unsqueeze(0)
                    side = side.unsqueeze(0)
                    _front = front
                    _side = side
                    gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=0.3)
                    front = torch.rot90(front, k=-2, dims=(3, 4))
                    side = torch.rot90(side, k=-2, dims=(3, 4))
                    temp =gaussian_blur(side[0])
                    side = temp.unsqueeze(0)

                    temp = gaussian_blur(front[0])  # 先模糊
                    front_for_grho = temp.unsqueeze(0)
                    if len(front_for_grho_List) < 2:
                        front_for_grho_List.append(front_for_grho)
                else:
                    front, side = dealrender(trueDen[0, 0],isSample=False,f=modelConfig['den_size'][0]) #[1,1,1,64,64]
                    front = torch.clamp(front,min=0,max=1)
                    side = torch.clamp(side,min=0,max=1)
                    _front = front
                    _side = side
                    front_for_grho = front
                    if len(front_for_grho_List) < 2:
                        front_for_grho_List.append(front_for_grho)
                    # front_for_grho.float().to(device)
                    front.float().to(device)
                    side.float().to(device)

                number = int(num[0])


                if len(denList) != 2:
                    # 预测
                    if len(frontList) < 2:
                        img = torch.rot90(front, k=2, dims=(3, 4)) #[1,1,1,64,64]
                        frontList.append(img)
                        trueSideList.append(side)
                    if len(frontList) == 2:
                        prompt = torch.cat(frontList,dim=2) #[1,1,2,64,64]
                        side = torch.cat(trueSideList,dim=2)
                        if modelConfig['usePredict']:
                            predict_side = InitPredict(prompt.float(), side.float(), predictInitModel, device, num[0], modelConfig,
                                                   savepath=savepath)  # [1,1,2,64,64] 正的

                            prompt_side = predict_side
                            ft_prompt_side = predict_side
                            side = torch.rot90(predict_side, k=-2, dims=(3, 4)) # [1,1,2,64,64]


                        denList = []
                        for idx,input in enumerate(front_for_grho_List): #前两帧的粗密度场
                            # 粗密度估计
                            with torch.no_grad():
                                imgList = []
                                for i in range(16):
                                    if i == 0:
                                        front_for_grho = F.interpolate(input[0], size=(modelConfig['img_size']), mode='bilinear',
                                                             align_corners=False).unsqueeze(0)
                                        imgList.append(front_for_grho)  # [1,1,1,64,64]
                                    elif i == 8:
                                        side_for_grho = F.interpolate(side[0,:,idx:idx+1,:,:], size=(modelConfig['img_size']), mode='bilinear',
                                                         align_corners=False).unsqueeze(0)
                                        imgList.append(side_for_grho)
                                    else:
                                        img = torch.zeros(size=[1, 1, 1, modelConfig['img_size'][0], modelConfig['img_size'][1]], device=device)
                                        imgList.append(img)
                                imgs = torch.cat(imgList,dim=2)
                                den, _, _, _, _ = denModel(imgs.to(device))
                                den.to(device)
                                denList.append(den)

                                den_np = den[0].detach().cpu().numpy()
                                np.savez(f'{savepath}/coarseDensity_{number-1+idx:06d}.npz', array=den_np)

                        # '''
                        # 细密度估计 view = 4
                        # '''
                        # # 超分
                        # if use_img_num != 2:
                        #     all_prompt_images = []
                        #     # for i in angle_idx:
                        #
                        #     for i in [4,12]:
                        #         angle = 180.0 / 16 * i + 90
                        #         _, _, z0 = render(denList[0][0, 0], angle=angle,
                        #                           f=modelConfig['den_size'][0])  # [1,1,64,64]
                        #         _, _, z1 = render(denList[1][0, 0], angle=angle,
                        #                           f=modelConfig['den_size'][0])  # [1,1,64,64]
                        #         promptImage = torch.cat((z0, z1), dim=1)
                        #         promptImage = promptImage.unsqueeze(1)  # [1,1,9,64,64]
                        #         all_prompt_images.append(promptImage)
                        #     all_prompt_images = torch.cat(all_prompt_images, dim=0)  # [b,1,2,64,64]
                        #     z_H_0 = init_hr(all_prompt_images.to(device), hrInitModel, device, num[0], modelConfig,
                        #                     savepath=savepath, angle_idx=[4,12])  # [b,1,2,64,64]
                        #     z_H_0 = torch.clamp(z_H_0, min=0)
                        #     # prompt_z_H_0 = z_H_0
                        #
                        # for idx,den_c in enumerate(denList):
                        #     imgList = []
                        #     temp_z_H_0 = z_H_0[:,:,idx:idx+1,:,:] # [b,1,1,64,64]
                        #     for i in range(16):
                        #         if i == 0:
                        #             front_for_grho = F.interpolate(front_for_grho_List[idx][0],
                        #                                            size=(modelConfig['img_size']), mode='bilinear',
                        #                                            align_corners=False).unsqueeze(0)
                        #             imgList.append(front_for_grho)  # [1,1,1,64,64]
                        #         elif i == 8:
                        #             # x = torch.rot90(x.unsqueeze(0), k=2, dims=(3, 4))
                        #             side_for_grho = F.interpolate(side[0, :, idx:idx + 1, :, :],
                        #                                           size=(modelConfig['img_size']), mode='bilinear',
                        #                                           align_corners=False).unsqueeze(0)
                        #             imgList.append(side_for_grho)
                        #         # elif i not in angle_idx:
                        #         elif i not in [4,12]:
                        #             angle = 180.0 / 16 * i + 90
                        #             _, _, z = render(den_c[0, 0], angle=angle,
                        #                              f=modelConfig['den_size'][0])  # [1,1,64,64]
                        #             img = torch.rot90(z.unsqueeze(0), k=2, dims=(3, 4))
                        #             # img = torch.zeros(size=[1, 1, 1, modelConfig['img_size'][0], modelConfig['img_size'][1]], device=device)
                        #
                        #             imgList.append(img)
                        #         elif i in [4,12]:
                        #             temp = torch.rot90(temp_z_H_0[0:1, 0, :, :, :].unsqueeze(0), k=2, dims=(3, 4))
                        #             imgList.append(temp)
                        #             if temp_z_H_0.shape[0] != 1:
                        #                 temp_z_H_0 = temp_z_H_0[1:,:, :, :, :]
                        #     imgs = torch.cat(imgList, dim=2)  # [1,1,16,64,64]
                        #
                        #     with torch.no_grad():
                        #         den, _, _, _, _ = denMulModel(imgs.to(device))
                        #     denList[idx] = den.detach()

                        # '''
                        # 细密度估计 view = 8
                        # '''
                        # # 超分
                        # if use_img_num != 2:
                        #     all_prompt_images = []
                        #     # for i in angle_idx:
                        #
                        #     for i in [2, 4, 6, 10, 12, 14]:
                        #         angle = 180.0 / 16 * i + 90
                        #         _, _, z0 = render(denList[0][0, 0], angle=angle,
                        #                           f=modelConfig['den_size'][0])  # [1,1,64,64]
                        #         _, _, z1 = render(denList[1][0, 0], angle=angle,
                        #                           f=modelConfig['den_size'][0])  # [1,1,64,64]
                        #         promptImage = torch.cat((z0, z1), dim=1)
                        #         promptImage = promptImage.unsqueeze(1)  # [1,1,9,64,64]
                        #         all_prompt_images.append(promptImage)
                        #     all_prompt_images = torch.cat(all_prompt_images, dim=0)  # [b,1,2,64,64]
                        #     z_H_0 = init_hr(all_prompt_images.to(device), hrInitModel, device, num[0], modelConfig,
                        #                     savepath=savepath, angle_idx=[2, 4, 6, 10, 12, 14])  # [b,1,2,64,64]
                        #     z_H_0 = torch.clamp(z_H_0, min=0)
                        #     # prompt_z_H_0 = z_H_0
                        #
                        # for idx, den_c in enumerate(denList):
                        #     imgList = []
                        #     temp_z_H_0 = z_H_0[:, :, idx:idx + 1, :, :]  # [b,1,1,64,64]
                        #     for i in range(16):
                        #         if i == 0:
                        #             front_for_grho = F.interpolate(front_for_grho_List[idx][0],
                        #                                            size=(modelConfig['img_size']), mode='bilinear',
                        #                                            align_corners=False).unsqueeze(0)
                        #             imgList.append(front_for_grho)  # [1,1,1,64,64]
                        #         elif i == 8:
                        #             # x = torch.rot90(x.unsqueeze(0), k=2, dims=(3, 4))
                        #             side_for_grho = F.interpolate(side[0, :, idx:idx + 1, :, :],
                        #                                           size=(modelConfig['img_size']), mode='bilinear',
                        #                                           align_corners=False).unsqueeze(0)
                        #             imgList.append(side_for_grho)
                        #         # elif i not in angle_idx:
                        #         elif i not in [2, 4, 6, 10, 12, 14]:
                        #             angle = 180.0 / 16 * i + 90
                        #             _, _, z = render(den_c[0, 0], angle=angle,
                        #                              f=modelConfig['den_size'][0])  # [1,1,64,64]
                        #             img = torch.rot90(z.unsqueeze(0), k=2, dims=(3, 4))
                        #             # img = torch.zeros(size=[1, 1, 1, modelConfig['img_size'][0], modelConfig['img_size'][1]], device=device)
                        #
                        #             imgList.append(img)
                        #         elif i in [2, 4, 6, 10, 12, 14]:
                        #             temp = torch.rot90(temp_z_H_0[0:1, 0, :, :, :].unsqueeze(0), k=2, dims=(3, 4))
                        #             imgList.append(temp)
                        #             if temp_z_H_0.shape[0] != 1:
                        #                 temp_z_H_0 = temp_z_H_0[1:, :, :, :, :]
                        #     imgs = torch.cat(imgList, dim=2)  # [1,1,16,64,64]
                        #
                        #     with torch.no_grad():
                        #         den, _, _, _, _ = denMulModel(imgs.to(device))
                        #     denList[idx] = den


                        '''
                        细密度估计 view = 16
                        '''

                        #超分
                        if use_img_num != 2:
                            all_prompt_images = []
                            # for i in angle_idx:

                            for i in sr_idx:
                                angle = 180.0/16 * i + 90
                                _, _, z0 = render(denList[0][0, 0],angle=angle,f=modelConfig['den_size'][0])  # [1,1,64,64]
                                _, _, z1 = render(denList[1][0, 0], angle=angle,
                                                  f=modelConfig['den_size'][0])  # [1,1,64,64]
                                promptImage = torch.cat((z0,z1), dim=1)
                                promptImage = promptImage.unsqueeze(1)  # [1,1,9,64,64]
                                all_prompt_images.append(promptImage)
                            all_prompt_images = torch.cat(all_prompt_images, dim=0)  # [b,1,2,64,64]
                            z_H_0 = init_hr(all_prompt_images.to(device), hrInitModel, device, num[0], modelConfig,
                                           savepath=savepath, angle_idx=sr_idx)  # [b,1,2,64,64]
                            z_H_0 = torch.clamp(z_H_0, min=0)
                            for xiabiao,i in enumerate(sr_idx):
                                prev_prompt_dict[i] = z_H_0[xiabiao,:,0,:,:]
                                prompt_dict[i] = z_H_0[xiabiao,:,1,:,:]
                            # prompt_z_H_0 = z_H_0

                            _, _, z0 = render(denList[0][0, 0], angle=180, f=modelConfig['den_size'][0])  # [1,1,64,64]
                            _, _, z1 = render(denList[1][0, 0], angle=180, f=modelConfig['den_size'][0])  # [1,1,64,64]

                            #细密度估计
                            for idx,den_c in enumerate(denList):
                                imgList = []
                                # temp_z_H_0 = z_H_0[:,:,idx:idx+1,:,:] # [b,1,1,64,64]
                                for i in range(16):
                                    if i == 0:
                                        front_for_grho = F.interpolate(front_for_grho_List[idx][0],
                                                                       size=(modelConfig['img_size']), mode='bilinear',
                                                                       align_corners=False).unsqueeze(0)
                                        imgList.append(front_for_grho)  # [1,1,1,64,64]
                                    elif i == 8:
                                        # x = torch.rot90(x.unsqueeze(0), k=2, dims=(3, 4))
                                        side_for_grho = F.interpolate(side[0, :, idx:idx + 1, :, :],
                                                                      size=(modelConfig['img_size']), mode='bilinear',
                                                                      align_corners=False).unsqueeze(0)
                                        imgList.append(side_for_grho)
                                    # elif i not in angle_idx:
                                    elif i not in angle_idx:
                                        angle = 180.0 / 16 * i + 90
                                        _, _, z = render(den_c[0, 0], angle=angle,
                                                         f=modelConfig['den_size'][0])  # [1,1,64,64]
                                        img = torch.rot90(z.unsqueeze(0), k=2, dims=(3, 4))
                                        # img = torch.zeros(size=[1, 1, 1, modelConfig['img_size'][0], modelConfig['img_size'][1]], device=device)

                                        imgList.append(img)
                                    elif i in angle_idx:
                                        # temp = torch.rot90(temp_z_H_0[0:1, 0, :, :, :].unsqueeze(0), k=2, dims=(3, 4))
                                        if idx == 0:
                                            temp = torch.rot90(prev_prompt_dict[i].unsqueeze(0).unsqueeze(0), k=2, dims=(3, 4))
                                        else:
                                            temp = torch.rot90(prompt_dict[i].unsqueeze(0).unsqueeze(0), k=2,
                                                               dims=(3, 4))
                                        imgList.append(temp)
                                        # if temp_z_H_0.shape[0] != 1:
                                        #     temp_z_H_0 = temp_z_H_0[1:,:, :, :, :]
                                imgs = torch.cat(imgList, dim=2)  # [1,1,16,64,64]

                                with torch.no_grad():
                                    den, _, _, _, _ = denMulModel(imgs.to(device))
                                denList[idx] = den.detach()

                                render_front, render_side = dealrender(den[0][0], isSample=True,needrot=True,f=modelConfig['den_size'][0])

                                if modelConfig['dataset'] == "scalarflow":
                                    trueFront = frontList[idx]
                                    trueSide = trueSideList[idx] #[1,1,1,64,64]
                                    trueSide = torch.rot90(trueSide,k=2,dims=[3,4])
                                    trueFront = F.interpolate(trueFront[0], size=(modelConfig['img_size']),
                                                              mode='bilinear',
                                                              align_corners=False).unsqueeze(0)
                                    trueSide = F.interpolate(trueSide[0], size=(modelConfig['img_size']),
                                                             mode='bilinear',
                                                             align_corners=False).unsqueeze(0)

                                else:
                                    trueFront = frontList[idx]
                                    trueSide = trueSideList[idx]  # [1,1,1,64,64]
                                    trueSide = torch.rot90(trueSide, k=2, dims=[3, 4])
                                if modelConfig['dataset'] == "scalarflow":
                                    combined = torch.cat([render_front, trueFront, render_side, trueSide], dim=-1) / 2.0
                                else:
                                    combined = torch.cat([render_front, trueFront, render_side, trueSide], dim=-1)
                                combined = F.interpolate(combined[0], size=(4*modelConfig['img_size'][0], 4*modelConfig['img_size'][1]*4), mode='bilinear',
                                                         align_corners=False).squeeze(0)
                                save_image(combined, os.path.join(savepath, f"woAdvected{number-1+idx:06d}.png"), nrow=1)
                                images = renderEightViews(den[0][0],f=modelConfig['den_size'][0])

                                if modelConfig['dataset'] == "scalarflow":
                                    images = torch.cat(images, dim=-1) / 2.0
                                else:
                                    images = torch.cat(images, dim=-1)
                                images = F.interpolate(images.unsqueeze(0), size=(4*modelConfig['img_size'][0], 4*modelConfig['img_size'][1]*8), mode='bilinear',
                                                       align_corners=False).squeeze(0)
                                save_image(images, os.path.join(savepath, f"woAdvectedEightViews{number-1+idx:06d}.png"),
                                           nrow=1)
                                epoch_den_loss += F.mse_loss(den, trueDen)
                                logger.info(f'{number-1+idx:06d} frame density loss:{F.mse_loss(den, trueDen)}')


                else:
                    torch.cuda.empty_cache()
                    if modelConfig['useGrho']:
                        # 预测侧视图
                        if modelConfig['usePredict']:
                            prompt_front = torch.rot90(front, k=2, dims=(3, 4))#[:, :, 0:1, :, :]  # [1,1,1,64,64]
                            # prompt_front = torch.rot90(prompt_front, k=2, dims=(3, 4))# / 6.0  # * 5  # [1,1,1,64,64]
                            prompt = torch.cat((prompt_side, prompt_front), dim=2)
                            ft_prompt = torch.cat((ft_prompt_side, prompt_front), dim=2)
                            # print(predict_side.shape)
                            dmpic,predict_side = predict(prompt.float(), side.float(), predictModel, device, num[0],
                                                   modelConfig,savepath=savepath,ft_prompt = ft_prompt,frame=count)  # [1,1,3,64,64]
                            # predict_side = predict_side[:, :, 0:1, :, :]  # [1,1,1,64,64] 正的
                            dmpic = dmpic[:, :, 0:1, :, :]
                            side = torch.rot90(predict_side, k=-2, dims=(3, 4)) #* 6.0  # / 5.0  # [1,1,1,64,64]
                            # side = side.squeeze(0).unsqueeze(-1).repeat(1, 1, 1, 1, 64)  # [1,1,64,64,64]
                            prompt_side = torch.cat((prompt_side[:, :, 1:, :, :], dmpic), dim=2)
                        with torch.no_grad():

                            front_for_grho = F.interpolate(front_for_grho[0], size=modelConfig['img_size'],
                                                          mode='bilinear', align_corners=False).unsqueeze(0)
                            side = F.interpolate(side[0], size=modelConfig['img_size'], mode='bilinear',
                                                         align_corners=False).unsqueeze(0)
                            imgs = torch.zeros((1, 1, 16, modelConfig['img_size'][0], modelConfig['img_size'][1]),
                                               device=device)
                            imgs[:, :, 0] = front_for_grho
                            imgs[:, :, 8] = side

                            den, _, _, _, _ = denModel(imgs.to(device))
                            den.to(device)

                            den_np = den[0].detach().cpu().numpy()
                            np.savez(f'{savepath}/coarseDensity_{number:06d}.npz', array=den_np)

                            _, _, z = render(den[0, 0], angle=180, f=modelConfig['den_size'][0],
                                             SF_flag=SFflag)  # [1,1,64,64]
                            z = F.interpolate(z, size=(64, 64), mode='bilinear', align_corners=False)

                            if modelConfig['usePredict']:
                                ft_prompt_side = torch.cat((ft_prompt_side[:, :, 1:2, :, :], z.unsqueeze(0)), dim=2)



                            '''
                            view = 4
                            '''
                            # 超分
                            if use_img_num != 2:
                                all_cur = []
                                # for i in angle_idx:

                                for i in view_dict[4]:
                                    angle = 180.0 / 16 * i + 90
                                    x, y, z = render(den[0, 0], angle=angle,
                                                     f=modelConfig['den_size'][0])  # [1,1,64,64]
                                    cur = torch.cat((x, y, z), dim=1)  # [1,3,64,64]
                                    all_cur.append(cur)
                                all_cur = torch.cat(all_cur, dim=0).unsqueeze(1)  # [b,1,3,64,64]
                                all_prompt_images = []
                                temp = all_cur
                                for i in view_dict[4]:
                                    prev0 = prev_prompt_dict[i].unsqueeze(0).unsqueeze(0)
                                    prev1 = prompt_dict[i].unsqueeze(0).unsqueeze(0)
                                    prev = torch.cat((prev0,prev1),dim=2).repeat_interleave(3, dim=2)  # [1,1,6,64,64]
                                    prev = F.interpolate(prev[0], scale_factor=0.5, mode='bilinear',
                                                         align_corners=False).unsqueeze(0)
                                    prev = F.interpolate(prev[0], modelConfig['img_size'], mode='bilinear',
                                                         align_corners=False).unsqueeze(0)
                                    promptImage = torch.cat((temp[0:1], prev), dim=2)  # [1,1,9,64,64]
                                    if temp.shape[0] != 1:
                                        temp = temp[1:]
                                    all_prompt_images.append(promptImage)
                                all_prompt_images = torch.cat(all_prompt_images, dim=0)  # [b,1,9,64,64]

                                z_H_0 = hr_test(all_prompt_images.to(device), hrModel, device, num[0], modelConfig,
                                               savepath=savepath, angle_idx=view_dict[4])  # [b,1,64,64]
                                z_H_0 = torch.clamp(z_H_0, min=0)
                                for xiabiao, i in enumerate(view_dict[4]):
                                    now_sr_dict[i] = z_H_0[xiabiao]



                                imgList = []
                                for i in range(16):
                                    if i == 0:
                                        imgList.append(front_for_grho)  # [1,1,1,64,64]
                                    elif i == 8:
                                        imgList.append(side)
                                    elif i not in view_dict[4]:
                                        angle = 180.0 / 16 * i + 90
                                        _, _, z = render(den[0, 0], angle=angle,
                                                         f=modelConfig['den_size'][0])  # [1,1,1,64,64]
                                        img = torch.rot90(z.unsqueeze(0), k=-2, dims=(3, 4))
                                        imgList.append(img)
                                    elif i in view_dict[4]:
                                        temp = torch.rot90(now_sr_dict[i].unsqueeze(0).unsqueeze(0), k=2, dims=(3, 4))
                                        imgList.append(temp)
                                imgs = torch.cat(imgList, dim=2)  # [1,1,16,64,64]

                                with torch.no_grad():
                                    den, _, _, _, _ = denMulModel(imgs.to(device))
                                    den.to(device)

                            '''
                            view = 8
                            '''
                            # 超分
                            if use_img_num != 2:
                                all_cur = []
                                # for i in angle_idx:

                                for i in view_dict[8]:
                                    angle = 180.0 / 16 * i + 90
                                    x, y, z = render(den[0, 0], angle=angle,
                                                     f=modelConfig['den_size'][0])  # [1,1,64,64]
                                    cur = torch.cat((x, y, z), dim=1)  # [1,3,64,64]
                                    all_cur.append(cur)
                                all_cur = torch.cat(all_cur, dim=0).unsqueeze(1)  # [b,1,3,64,64]
                                all_prompt_images = []
                                temp = all_cur
                                for i in view_dict[8]:
                                    prev0 = prev_prompt_dict[i].unsqueeze(0).unsqueeze(0)
                                    prev1 = prompt_dict[i].unsqueeze(0).unsqueeze(0)
                                    prev = torch.cat((prev0, prev1), dim=2).repeat_interleave(3, dim=2)  # [1,1,6,64,64]
                                    prev = F.interpolate(prev[0], scale_factor=0.5, mode='bilinear',
                                                         align_corners=False).unsqueeze(0)
                                    prev = F.interpolate(prev[0], modelConfig['img_size'], mode='bilinear',
                                                         align_corners=False).unsqueeze(0)
                                    promptImage = torch.cat((temp[0:1], prev), dim=2)  # [1,1,9,64,64]
                                    if temp.shape[0] != 1:
                                        temp = temp[1:]
                                    all_prompt_images.append(promptImage)
                                all_prompt_images = torch.cat(all_prompt_images, dim=0)  # [b,1,9,64,64]

                                z_H_0 = hr_test(all_prompt_images.to(device), hrModel, device, num[0], modelConfig,
                                               savepath=savepath, angle_idx=view_dict[8])  # [b,1,64,64]
                                z_H_0 = torch.clamp(z_H_0, min=0)
                                for i,idx in enumerate(view_dict[8]):
                                    now_sr_dict[idx] = z_H_0[i]

                                imgList = []
                                for i in range(16):
                                    if i == 0:
                                        imgList.append(front_for_grho)  # [1,1,1,64,64]
                                    elif i == 8:
                                        imgList.append(side)
                                    elif i not in view_dict[8] and i not in view_dict[4]:
                                        angle = 180.0 / 16 * i + 90
                                        _, _, z = render(den[0, 0], angle=angle,
                                                         f=modelConfig['den_size'][0])  # [1,1,1,64,64]
                                        img = torch.rot90(z.unsqueeze(0), k=-2, dims=(3, 4))
                                        imgList.append(img)
                                    elif i in view_dict[8] or i in view_dict[4]:
                                        temp = torch.rot90(now_sr_dict[i].unsqueeze(0).unsqueeze(0), k=2, dims=(3, 4))
                                        imgList.append(temp)
                                imgs = torch.cat(imgList, dim=2)  # [1,1,16,64,64]

                                with torch.no_grad():
                                    den, _, _, _, _ = denMulModel(imgs.to(device))
                                    den.to(device)

                            '''
                            view = 16
                            '''

                            # 超分
                            if use_img_num != 2:
                                all_cur = []
                                all_prompt_images = []
                                for i in view_dict[16]:
                                    angle = 180.0 / 16 * i + 90
                                    x, y, z = render(den[0, 0], angle=angle,f=modelConfig['den_size'][0],SF_flag=SFflag)  # [1,1,64,64]
                                    cur = torch.cat((x, y, z), dim=1).unsqueeze(0) # [1,1,3,64,64]
                                    prev0 = prev_prompt_dict[i].unsqueeze(0).unsqueeze(0)
                                    prev1 = prompt_dict[i].unsqueeze(0).unsqueeze(0)
                                    prev = torch.cat((prev0, prev1), dim=2).repeat_interleave(3, dim=2)  # [1,1,6,64,64]
                                    # prev = torch.cat((prev0, prev1), dim=2).repeat(1,1,3,1,1)  # [1,1,6,64,64]
                                    prev = F.interpolate(prev[0], scale_factor=0.5, mode='bilinear', align_corners=False).unsqueeze(0)
                                    prev = F.interpolate(prev[0], modelConfig['img_size'], mode='bilinear', align_corners=False).unsqueeze(0)
                                    promptImage = torch.cat((cur, prev), dim=2)  # [1,1,9,64,64]
                                    all_prompt_images.append(promptImage)
                                all_prompt_images = torch.cat(all_prompt_images,dim=0) # [b,1,9,64,64]
                                z_H_0 = hr_test(all_prompt_images.to(device), hrModel, device, num[0], modelConfig,
                                               savepath=savepath, angle_idx=view_dict[16])  # [b,1,64,64]
                                z_H_0 = torch.clamp(z_H_0, min=0)
                                for i,idx in enumerate(view_dict[16]):
                                    now_sr_dict[idx] = z_H_0[i]

                                prev_prompt_dict = prompt_dict
                                prompt_dict = now_sr_dict


                                imgList = []
                                for i in range(16):
                                    if i == 0:
                                        imgList.append(front_for_grho)  # [1,1,1,64,64]
                                    elif i == 8:
                                        imgList.append(side)
                                    elif i not in sr_idx:
                                        angle = 180.0 / 16 * i + 90

                                        _, _, z = render(den[0, 0], angle=angle,
                                                         f=modelConfig['den_size'][0],SF_flag=SFflag)  # [1,1,1,64,64]
                                        img = torch.rot90(z.unsqueeze(0), k=-2, dims=(3, 4))
                                        imgList.append(img)
                                    elif i in sr_idx:
                                        temp = torch.rot90(now_sr_dict[i].unsqueeze(0).unsqueeze(0), k=2, dims=(3, 4))
                                        imgList.append(temp)
                                        if z_H_0.shape[0] != 1:
                                            z_H_0 = z_H_0[1:, :, :, :]
                                imgs = torch.cat(imgList, dim=2)  # [1,1,16,64,64]

                                with torch.no_grad():
                                    den, _, _, _, _ = denMulModel(imgs.to(device))
                                    den.to(device)


                            render_front, render_side = dealrender(den[0][0], isSample=True,needrot = True,f=modelConfig['den_size'][0])

                            if modelConfig['dataset'] == "scalarflow":
                                trueFront = _front
                                trueSide = _side
                                trueFront = F.interpolate(trueFront[0], size=(modelConfig['img_size']),
                                                               mode='bilinear',
                                                               align_corners=False).unsqueeze(0)
                                trueSide = F.interpolate(trueSide[0], size=(modelConfig['img_size']),
                                                          mode='bilinear',
                                                          align_corners=False).unsqueeze(0)
                            else:
                                trueFront, trueSide = dealrender(trueDen[0][0], isSample=False, needrot=True,f=modelConfig['den_size'][0])


                            if modelConfig['dataset'] == "scalarflow":
                                combined = torch.cat([render_front, trueFront, render_side, trueSide], dim=-1) / 2.0
                            else:
                                combined = torch.cat([render_front, trueFront, render_side, trueSide], dim=-1)  # / 2.0
                            combined = F.interpolate(combined[0], size=(4*modelConfig['img_size'][0], 4*modelConfig['img_size'][1]*4), mode='bilinear',
                                                     align_corners=False).squeeze(0)
                            save_image(combined, os.path.join(savepath, f"woAdvected{number:06d}.png"), nrow=1)
                            images = renderEightViews(den[0][0],f=modelConfig['den_size'][0])
                            if modelConfig['dataset'] == "scalarflow":
                                images = torch.cat(images, dim=-1) / 2.0
                            else:
                                images = torch.cat(images, dim=-1)  # / 2.0
                            images = F.interpolate(images.unsqueeze(0), size=(4*modelConfig['img_size'][0], 4*modelConfig['img_size'][1]*8), mode='bilinear',
                                                   align_corners=False).squeeze(0)
                            save_image(images, os.path.join(savepath, f"woAdvectedEightViews{number:06d}.png"),
                                       nrow=1)
                            epoch_den_loss += F.mse_loss(den, trueDen)
                            logger.info(f'{number:06d} frame density loss:{F.mse_loss(den, trueDen)}')
                    else:
                        den = trueDen

                    denList[0] = next_den.detach()
                    # denList[0] = denList[1]
                    denList[1] = den.detach()



                if len(denList) == 2:
                    flag = True

                if flag :
                    with torch.no_grad():
                        allDensity = torch.cat((denList[0].to(device), denList[1].to(device)), dim=1)
                        base_vel, _, _, _, _ = velModel(allDensity)
                        base_vel = set_vel_zero_border(base_vel)
                        base_vel.to(device)
                        vel = base_vel
                        logger.info(f'{number:06d} frame vel loss:{F.mse_loss(vel, trueVel)}')
                        epoch_vel_loss += F.mse_loss(vel, trueVel)

                    if is_optsource:
                        pass

                    else:
                        opt = torch.optim.AdamW([source], lr=1e-1)
                        scheduler = StepLR(opt, step_size=50, gamma=0.1)
                        cur_source = iteration(opt, source, denList[0][0, 0].detach(), denList[1].detach(), vel[0], t=150,
                                               scheduler=scheduler,need_print=True,config=modelConfig['den_size'][1])
                        mask = torch.zeros_like(cur_source, dtype=torch.bool)
                        mask[:,modelConfig['den_size'][1]//4:,:] = True
                        cur_source[mask] = 0
                        cur_source[cur_source < 1e-3] = 0
                        cur_source = set_zero_border(cur_source)


                    if is_optsource:
                        pass

                    else:
                        next_den = advection(denList[0][0, 0], vel[0], cur_source)
                        next_den = set_zero_border(next_den).unsqueeze(0).unsqueeze(0).to(device)  # torch.Size([1, 1, 64, 64, 64])
                        epoch_advected_den_loss = epoch_advected_den_loss + F.mse_loss(next_den, denList[1])


                    if not is_optsource:

                        vel_np = vel[0].cpu().numpy()
                        np.savez(f'{savepath}/velocity_{number - 1:06d}.npz', array=vel_np)
                        den_np = next_den[0].detach().cpu().numpy()
                        np.savez(f'{savepath}/AdvectedDensity_{number:06d}.npz', array=den_np)
                        den_np = denList[1].cpu().numpy()
                        np.savez(f'{savepath}/woAdvectedDensity_{number:06d}.npz', array=den_np)
                        src_np = cur_source.cpu().numpy()
                        np.savez(f'{savepath}/source_{number - 1:06d}.npz', array=src_np)
                        if isFirst:
                            den_np = denList[0].detach().cpu().numpy()
                            np.savez(f'{savepath}/woAdvectedDensity_{number - 1:06d}.npz', array=den_np)
                            np.savez(f'{savepath}/AdvectedDensity_{number - 1:06d}.npz', array=den_np)
                            isFirst = False

                        logger.info(f'{number:06d} frame advected loss:{F.mse_loss(next_den, denList[1])}')

                        front, side = dealrender(next_den[0][0], isSample=True,needrot=True,f=modelConfig['den_size'][0])
                        if modelConfig['dataset'] == "scalarflow":
                            trueFront = _front
                            trueSide = _side
                            trueFront = F.interpolate(trueFront[0], size=(modelConfig['img_size']),
                                                      mode='bilinear',
                                                      align_corners=False).unsqueeze(0)
                            trueSide = F.interpolate(trueSide[0], size=(modelConfig['img_size']),
                                                     mode='bilinear',
                                                     align_corners=False).unsqueeze(0)
                        else:
                            trueFront, trueSide = dealrender(trueDen[0][0], isSample=False,needrot=True,f=modelConfig['den_size'][0])



                        if modelConfig['dataset'] == "scalarflow":
                            combined = torch.cat([front, trueFront, side, trueSide], dim=-1) / 2.0
                        else:
                            combined = torch.cat([front, trueFront, side, trueSide], dim=-1)  # / 2.0

                        combined = F.interpolate(combined[0], size=(4*modelConfig['img_size'][0], 4*modelConfig['img_size'][1]*4), mode='bilinear',
                                                 align_corners=False).squeeze(0)

                        save_image(combined, os.path.join(savepath, f"Advected{number:06d}.png"), nrow=1)
                        images = renderEightViews(next_den[0][0],f=modelConfig['den_size'][0])
                        if modelConfig['dataset'] == "scalarflow":
                            images = torch.cat(images, dim=-1) / 2.0
                        else:
                            images = torch.cat(images, dim=-1)  # / 2.0
                        images = F.interpolate(images.unsqueeze(0), size=(4*modelConfig['img_size'][0], 4*modelConfig['img_size'][1]*8), mode='bilinear',
                                                       align_corners=False).squeeze(0)
                        save_image(images, os.path.join(savepath, f"AdvectedEightViews{number:06d}.png"), nrow=1)

            if not is_optsource :
                avg_epoch_advected_den_loss = epoch_advected_den_loss / (count - 1)
                avg_epoch_den_loss = epoch_den_loss / count
                avg_epoch_vel_loss = epoch_vel_loss / (count - 1)

                logger.info(f'Average advected density loss: {avg_epoch_advected_den_loss}')
                logger.info(f'Average velocity loss: {avg_epoch_vel_loss}')
                logger.info(f'Average density loss: {avg_epoch_den_loss}')



    source = torch.rand(modelConfig['den_size'], requires_grad=True, device=device)

    logger.info("start reconstruction")
    dealsource(dataloader=dataloader,source=source.requires_grad_(),is_optsource=False,prev_prompt_dict=prev_prompt_dict,prompt_dict=prompt_dict,now_sr_dict=now_sr_dict,view_dict=view_dict)
    end_time = time.time()
    use_time = end_time-start_time
    hours = int(use_time // 3600)
    minutes = int((use_time % 3600) // 60)
    seconds = int(use_time % 60)

    logger.info(f"花费时间：{hours}小时 {minutes}分钟 {seconds}秒")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)