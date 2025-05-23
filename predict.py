import  torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image
import torch.nn as nn

class DDIMSampler(nn.Module):
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

    def forward(self,x, x_front,count=1):
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
                # xt_next = at_next.sqrt() * x0_t  + c2 * et

                xs.append(xt_next)

        return xs, x0_preds

def predict(images,true_image,model,device,num,modelconfig,savepath,ftmodel=None,ft_prompt = None,frame=None):
    savepath = f'{savepath}/predict'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    device = torch.device(device)

    with torch.no_grad():

        dm_cond = torch.cat((images[:, :, :1, :, :], ft_prompt[:, :, :1, :, :],
                             images[:, :, 1:2, :, :], ft_prompt[:, :, 1:2, :, :],
                             images[:, :, -1:, :, :]), dim=2)

        noisyImage = torch.randn_like(dm_cond, device=device)
        if modelconfig['dataset'] == 'scalarflow':
            sampler = DDIMSampler(
                model, 1e-4, 0.02, 1000,200).to(device)

            _,sampledImgs = sampler(noisyImage, dm_cond)
            sampledImgs = sampledImgs[-1]
            ft_img = sampledImgs[:,:,:1,:,:]


        else:
            sampler = DDIMSampler(
                model, 1e-4, 0.02, 1000, 100).to(device)
            _, sampledImgs = sampler(noisyImage, dm_cond)
            sampledImgs = sampledImgs[-1]
            ft_img = sampledImgs[:, :, :1, :, :]
        true_image = torch.rot90(true_image[0,0,:,:,:], k=2, dims=(1, 2))
        sample = sampledImgs[0,0,0:1,:,:]
        true = true_image
        true_front = images[0, 0, -1:, :, :]
        combine1 = torch.cat((images[0,0,0:1,:,:],images[0,0,1:2,:,:],sample,true,true_front),dim=2)
        combine2 = torch.cat((ft_prompt[0,0,0:1,:,:],ft_prompt[0,0,1:2,:,:],ft_img[0,0,:1,:,:],true,true_front),dim=2)
        combine= torch.cat((combine1,combine2),dim=1)
        combine = F.interpolate(combine.unsqueeze(0), size=(896,1280), mode='bilinear',
                                 align_corners=False).squeeze(0)
        save_image(combine, os.path.join(savepath, f"predictAndTrue{num}.png"), nrow=1)

        sample =  F.interpolate(sample.unsqueeze(0), scale_factor=4, mode='bilinear',
                                 align_corners=False).squeeze(0)
        save_image(sample, os.path.join(savepath, f"SampleSide{num}.png"), nrow=1)



        sampledImgs = torch.clamp(sampledImgs,min=0,max=1)
        ft_img = torch.clamp(ft_img,min=0,max=1)
    return sampledImgs,ft_img
def InitPredict(images,true_image,model,device,num,modelconfig,savepath):
    savepath = f'{savepath}/predict'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    device = torch.device(device)
    noisyImage = torch.randn_like(images, device=device)
    with torch.no_grad():
        if modelconfig['dataset'] == 'scalarflow':
            sampler = DDIMSampler(
                model, 1e-4, 0.02, 1000,400).to(device)
            sampledImgs,_ = sampler(noisyImage, images)
            sampledImgs = sampledImgs[-1] #[1, 1, 2, 64, 64]
        else:
            sampler = DDIMSampler(
                model, 1e-4, 0.02, 1000, 200).to(device)
            sampledImgs, _ = sampler(noisyImage, images)
            sampledImgs = sampledImgs[-1]
        true_side = torch.rot90(true_image[0,0,:,:,:], k=2, dims=(1, 2))#[2,64,64]
        sample = sampledImgs[0,0]#[2,64,64]
        true_front = images[0, 0]#[2,64,64]
        for i in range(2):
            combine = torch.cat((sample[i:i+1],true_side[i:i+1],true_front[i:i+1]),dim=2)
            combine = F.interpolate(combine.unsqueeze(0), size=(256, 768), mode='bilinear',
                                     align_corners=False).squeeze(0)
            save_image(combine, os.path.join(savepath, f"predictAndTrue{int(num)-1+i:06d}.png"), nrow=1)

            our_sample =  F.interpolate(sample[i:i+1].unsqueeze(0), size=(256, 256), mode='bilinear',
                                     align_corners=False).squeeze(0)
            save_image(our_sample, os.path.join(savepath, f"SampleSide{int(num)-1+i:06d}.png"), nrow=1)




    return sampledImgs
