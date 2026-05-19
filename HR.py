import torch.nn as nn
import  torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image

class DDIMHRSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, num_step):
        super().__init__()
        self.model = model
        self.T = T

        self.betas = torch.linspace(beta_1, beta_T, T).double()
        skip = T // num_step
        self.seq = range(0, T, skip)




    def compute_alpha(self,beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0).cuda()
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
        return a

    def guidance(self, x, y, frame):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            front = y[:, :, -1:, :, :]
            front = front.repeat(1, 1, 3, 1, 1)
            x_0_t = torch.mean(x_in, dim=-1)
            x_0_t = x_0_t.unsqueeze(-1)
            x_0_t = x_0_t.repeat(1,1,1,1,64)
            front = torch.mean(front, dim=-1)
            front = front.unsqueeze(-1)
            front = front.repeat(1, 1, 1, 1, 64)
            loss = F.mse_loss(x_0_t, front)
            loss.backward()
            x_in.grad = x_in.grad * 10 * (1 + min(frame/30.0 * 5,5))
            torch.clamp_(x_in.grad, min=-1.0, max=1.0)
            return x_in.grad
    def forward(self,x, x_front,frame=None):
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(self.seq[:-1])
            x0_preds = []
            xs = [x]

            for i, j in zip(reversed(self.seq), reversed(seq_next)):
                t = (torch.ones(n,dtype=torch.long) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = self.compute_alpha(self.betas, t.long())
                at_next = self.compute_alpha(self.betas, next_t.long())
                xt = xs[-1]

                et = self.model(xt.float(), t, x_front.float())
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t)
                c1 = (
                      1* ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()

                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

                xs.append(xt_next)

        return xs, x0_preds

def init_hr(images,model,device,num,modelconfig,savepath,angle_idx):
    savepath = f'{savepath}/sr'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    device = torch.device(device)
    noisyImage = torch.randn_like(images,device=device)
    with torch.no_grad():
        if modelconfig['dataset'] == 'scalarflow':
            sampler = DDIMHRSampler(
                model, 1e-4, 0.02, 1000,10).to(device)
            secimages = images.clone()

            _,sampledImgs = sampler(noisyImage, secimages)
            sampledImgs = sampledImgs[-1]



        else:

            sampler = DDIMHRSampler(
                model, 1e-4, 0.02, 1000, 10).to(device)
            _, sampledImgs = sampler(noisyImage, images)
            sampledImgs = sampledImgs[-1]


        sample = sampledImgs[:,0,:,:,:] #[b,2,64,64]
        z_L = images[:, 0, :, :, :] #[b,2,64,64]
        combine = torch.cat((sample,z_L),dim=-1) #[b,2,64,128]
        combine = F.interpolate(combine, size=(448, 512), mode='bilinear',
                                 align_corners=False)
        sample =  F.interpolate(sample, size=(448, 256), mode='bilinear',
                                 align_corners=False)

        for t in range(2):

            for i,idx in enumerate(angle_idx):

                save_image(combine[i:i+1,t,:,:], os.path.join(savepath, f"SRCombine{int(num)-1+t:06d}_angleIdx{idx:02d}.png"), nrow=1)
                save_image(sample[i:i+1,t,:,:], os.path.join(savepath, f"SRSample{int(num)-1+t:06d}_angleIdx{idx:02d}.png"), nrow=1)


    return sampledImgs #[b,1,2,64,64]



def hr_test(images,model,device,num,modelconfig,savepath,angle_idx,isX=False):
    savepath = f'{savepath}/sr'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    device = torch.device(device)
    noisyImage = torch.randn(size=[1, 1, 1, 112, 64], device=device)

    secimages = images.clone()



    with torch.no_grad():

        sampledImgs = model(secimages.float())




        sample = sampledImgs[:,0,0:1,:,:] + secimages[:, 0, 2:3, :, :]#[b,1,64,64]
        z_L = images[:, 0, 2:3, :, :] #[b,1,64,64]
        # last1 = images[:,0,5:6,:,:]
        last2 = images[:,0,8:9,:,:]
        combine = torch.cat((sample,z_L,last2),dim=-2) #[b,1,64,128]
        combine = F.interpolate(combine, scale_factor=4, mode='bilinear',
                                 align_corners=False)
        sample =  F.interpolate(sample, scale_factor=4, mode='bilinear',
                                 align_corners=False)



        image_list = []

        for i, idx in enumerate(angle_idx):
            current_image = combine[i:i + 1, 0, :, :]  # 假设我们只取第一个通道
            image_list.append(current_image)

        combined_image = torch.cat(image_list, dim=-1)

        if isX:

            save_image(combined_image, os.path.join(savepath, f"SRSide{int(num):06d}_.png"), nrow=len(angle_idx))
        else:
            save_image(combined_image, os.path.join(savepath, f"SRCombine{int(num):06d}_.png"), nrow=len(angle_idx))

    return sampledImgs[:,0,0:1,:,:] + images[:, 0, 2:3, :, :] #[b,1,64,64]