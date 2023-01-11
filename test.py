# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet
from varnet import *
import pathlib
import numpy as np
import torch.optim as optim
from fastmri.data import  mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import math

# %% data loader
from my_data import SliceDataset

nc = 16
nx = 384
ny = 396

def data_transform(kspace_noisy, kspace_clean, ncc_effect,sense_maps):
    # Transform the kspace to tensor format
    ncc_effect = transforms.to_tensor(ncc_effect)
    kspace_noisy = transforms.to_tensor(kspace_noisy)
    kspace_noisy = torch.cat((kspace_noisy[torch.arange(nc),:,:].unsqueeze(-1),kspace_noisy[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    kspace_clean = transforms.to_tensor(kspace_clean)
    kspace_clean = torch.cat((kspace_clean[torch.arange(nc),:,:].unsqueeze(-1),kspace_clean[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    sense_maps = transforms.to_tensor(sense_maps)
    sense_maps = torch.cat((sense_maps[torch.arange(nc),:,:].unsqueeze(-1),sense_maps[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    return kspace_noisy, kspace_clean, ncc_effect, sense_maps

test_data = SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_copy/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_clean/train/'),
    transform=data_transform,
    challenge='multicoil'
)

def KtoIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

def MtoIm(im):
    Im = fastmri.rss(fastmri.complex_abs(im),dim=1)
    return Im

# %% define loss
L1Loss = torch.nn.L1Loss()
L2Loss = torch.nn.MSELoss()

import scipy.special as ss

def NccLoss(x1,x2,ncc_effect):
    L = ncc_effect[0,:,:].squeeze()
    s2 = ncc_effect[1,:,:].squeeze()
    x = x1*x2/s2
    y = torch.sum(torch.square(x1)/(2*s2)-(torch.log(ss.ive(L-1,x))+x)+(L-1)*torch.log(x1))
    return y/torch.sum(torch.ones_like(x))

from pytorch_msssim import SSIM

ssim_loss = SSIM(data_range=100, size_average=True, channel=1)

# %% sampling mask
mask = torch.zeros(ny)
mask[torch.arange(99)*4] = 1
mask[torch.arange(186,210)] =1
mask = mask.bool().unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(nc,nx,1,2)


# %% imnet loader
epoch = 180
imunet = torch.load('/home/wjy/Project/refnoise_model/imunet_mse_acc4_epoch'+str(epoch),map_location=torch.device('cpu'))

# %%
with torch.no_grad():
    kspace_noisy, kspace_clean, ncc_effect, sense_maps = test_data[0]
    kspace_noisy = kspace_noisy.unsqueeze(0) 
    kspace_clean = kspace_clean.unsqueeze(0)
    gt = KtoIm(kspace_clean)
    gt_noise = KtoIm(kspace_noisy)

    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace_noisy,Mask)

    image = fastmri.ifft2c(kspace_undersample)
    image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
    image_output = imunet(image_input)
    recon = torch.cat((image_output[:,torch.arange(nc),:,:].unsqueeze(4),image_output[:,torch.arange(nc,2*nc),:,:].unsqueeze(4)),4)

    recon = MtoIm(recon)

# %%
sp = torch.ge(gt, 0.03*torch.max(gt))
print("MSE:",L2Loss(recon,gt))
#print("MSE roi:",L2Loss(torch.mul(recon,sp),torch.mul(gt,sp)))

print("MAE:",L1Loss(recon,gt))
#print("MAE roi:",L1Loss(torch.mul(recon,sp),torch.mul(gt,sp)))

print("MSE approx:",L2Loss(recon,gt_noise))
print("MAE approx:",L1Loss(recon,gt_noise))
print("NCE:", NccLoss(recon.squeeze(),gt_noise,ncc_effect)-NccLoss(gt_noise,gt_noise,ncc_effect))


# %%
test_count = 0
mse, mse_approx, mae, mae_approx, ssim, ssim_approx, nce = 0, 0, 0, 0, 0, 0, 0

for kspace_noisy, kspace_clean, ncc_effect, sense_maps in test_data:
    test_count += 1
    with torch.no_grad():
        kspace_noisy = kspace_noisy.unsqueeze(0) 
        kspace_clean = kspace_clean.unsqueeze(0)
        sense_maps = sense_maps.unsqueeze(0)
        gt = KtoIm(kspace_clean)
        gt_noise = KtoIm(kspace_noisy)

        # undersampling
        Mask = mask.unsqueeze(0)
        kspace_undersample = torch.mul(kspace_noisy,Mask)

        image = fastmri.ifft2c(kspace_undersample)
        image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
        image_output = imunet(image_input)
        recon = torch.cat((image_output[:,torch.arange(nc),:,:].unsqueeze(4),image_output[:,torch.arange(nc,2*nc),:,:].unsqueeze(4)),4)

        recon = MtoIm(recon)

        # evaluation
        mse += L2Loss(recon,gt)
        mse_approx += L2Loss(recon,gt_noise)
        mae += L1Loss(recon,gt)
        mae_approx += L1Loss(recon,gt_noise)
        ssim += 1-ssim_loss(recon.unsqueeze(0),gt.unsqueeze(0))
        ssim_approx += 1-ssim_loss(recon.unsqueeze(0),gt_noise.unsqueeze(0))
        nce += NccLoss(recon.squeeze(),gt_noise,ncc_effect)-NccLoss(gt_noise,gt_noise,ncc_effect)

print(mse/test_count,mse_approx/test_count,mae/test_count,mae_approx/test_count,ssim/test_count,ssim_approx/test_count,nce/test_count )

# %% varnet loader
epoch = 140 
sigma = 1
cascades = 12
chans = 16
varnet = torch.load("/home/wjy/Project/refnoise_model/varnet_mae_acc4_cascades"+str(cascades)+"_channels"+str(chans)+"_epoch"+str(epoch),map_location = 'cpu')

# %%
with torch.no_grad():
    kspace_noisy, kspace_clean, ncc_effect, sense_maps = test_data[0]
    kspace_noisy = kspace_noisy.unsqueeze(0) 
    kspace_clean = kspace_clean.unsqueeze(0)
    gt = KtoIm(kspace_clean)
    gt_noise = KtoIm(kspace_noisy)

    # undersampling
    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace_noisy,Mask)

    #reconstruction
    recon_M = varnet(kspace_undersample, Mask, 24)
    recon = MtoIm(recon_M)   

# %%
sp = torch.ge(gt, 0.03*torch.max(gt))
print("MSE:",L2Loss(recon,gt))
print("MSE roi:",L2Loss(torch.mul(recon,sp),torch.mul(gt,sp)))

print("MAE:",L1Loss(recon,gt))
print("MAE roi:",L1Loss(torch.mul(recon,sp),torch.mul(gt,sp)))

print("MSE approx:",L2Loss(recon,gt_noise))
print("MAE approx:",L1Loss(recon,gt_noise))
print("NCE:", NccLoss(recon.squeeze(),gt_noise,ncc_effect)-NccLoss(gt_noise,gt_noise,ncc_effect))

# %%
test_count = 0
mse, mse_approx, mae, mae_approx, ssim, ssim_approx, nce = 0, 0, 0, 0, 0, 0, 0

for kspace_noisy, kspace_clean, ncc_effect in test_data:
    test_count += 1
    with torch.no_grad():
        kspace_noisy = kspace_noisy.unsqueeze(0) 
        kspace_clean = kspace_clean.unsqueeze(0)
        gt = KtoIm(kspace_clean)
        gt_noise = KtoIm(kspace_noisy)

        # undersampling
        Mask = mask.unsqueeze(0)
        kspace_undersample = torch.mul(kspace_noisy,Mask)

        #reconstruction
        recon_M = varnet(kspace_undersample, Mask, 24)
        recon = MtoIm(recon_M)     

        # evaluation
        mse += L2Loss(recon,gt)
        mse_approx += L2Loss(recon,gt_noise)
        mae += L1Loss(recon,gt)
        mae_approx += L1Loss(recon,gt_noise)
        ssim += 1-ssim_loss(recon.unsqueeze(0),gt.unsqueeze(0))
        ssim_approx += 1-ssim_loss(recon.unsqueeze(0),gt_noise.unsqueeze(0))
        nce += NccLoss(recon.squeeze(),gt_noise,ncc_effect)-NccLoss(gt_noise,gt_noise,ncc_effect)

print(mse/test_count,mse_approx/test_count,mae/test_count,mae_approx/test_count,ssim/test_count,ssim_approx/test_count,nce/test_count )

# %% modl loader
epoch = 80
modl = torch.load("/home/wjy/Project/refnoise_model/modl_mae_acc4_epochs"+str(epoch),map_location = 'cpu')

# %%
with torch.no_grad():
    kspace_noisy, kspace_clean, ncc_effect, sense_maps = test_data[0]
    kspace_noisy = kspace_noisy.unsqueeze(0) 
    kspace_clean = kspace_clean.unsqueeze(0)
    sense_maps = sense_maps.unsqueeze(0)
    gt = KtoIm(kspace_clean)
    gt_noise = KtoIm(kspace_noisy)

    # undersampling
    Mask = mask.unsqueeze(0)

    #reconstruction
    image_zf = fastmri.complex_mul(fastmri.complex_conj(sense_maps),fastmri.ifft2c(torch.mul(Mask,kspace_noisy))) 
    image_zf = torch.permute(torch.sum(image_zf, dim=1),(0,3,1,2))
    sense_maps = torch.complex(sense_maps[:,:,:,:,0],sense_maps[:,:,:,:,1])
    recon = modl(image_zf, sense_maps, Mask[:,0,:,:,0].squeeze())
    recon = fastmri.complex_abs(torch.permute(recon,(0,2,3,1)))   


# %%
sp = torch.ge(gt, 0.03*torch.max(gt))
print("MSE:",L2Loss(recon,gt))
#print("MSE roi:",L2Loss(torch.mul(recon,sp),torch.mul(gt,sp)))

print("MAE:",L1Loss(recon,gt))
#print("MAE roi:",L1Loss(torch.mul(recon,sp),torch.mul(gt,sp)))

print("MSE approx:",L2Loss(recon,gt_noise))
print("MAE approx:",L1Loss(recon,gt_noise))
print("NCE:", NccLoss(recon.squeeze(),gt_noise,ncc_effect)-NccLoss(gt_noise,gt_noise,ncc_effect))


# %%
# %%
test_count = 0
mse, mse_approx, mae, mae_approx, ssim, ssim_approx, nce = 0, 0, 0, 0, 0, 0, 0

for kspace_noisy, kspace_clean, ncc_effect, sense_maps in test_data:
    test_count += 1
    with torch.no_grad():
        kspace_noisy = kspace_noisy.unsqueeze(0) 
        kspace_clean = kspace_clean.unsqueeze(0)
        sense_maps = sense_maps.unsqueeze(0)
        gt = KtoIm(kspace_clean)
        gt_noise = KtoIm(kspace_noisy)

        # undersampling
        Mask = mask.unsqueeze(0)
        #reconstruction
        image_zf = fastmri.complex_mul(fastmri.complex_conj(sense_maps),fastmri.ifft2c(torch.mul(Mask,kspace_noisy))) 
        image_zf = torch.permute(torch.sum(image_zf, dim=1),(0,3,1,2))
        sense_maps = torch.complex(sense_maps[:,:,:,:,0],sense_maps[:,:,:,:,1])
        recon = modl(image_zf, sense_maps, Mask[:,0,:,:,0].squeeze())
        recon = fastmri.complex_abs(torch.permute(recon,(0,2,3,1)))   

        # evaluation
        mse += L2Loss(recon,gt)
        mse_approx += L2Loss(recon,gt_noise)
        mae += L1Loss(recon,gt)
        mae_approx += L1Loss(recon,gt_noise)
        ssim += 1-ssim_loss(recon.unsqueeze(0),gt.unsqueeze(0))
        ssim_approx += 1-ssim_loss(recon.unsqueeze(0),gt_noise.unsqueeze(0))
        nce += NccLoss(recon.squeeze(),gt_noise,ncc_effect)-NccLoss(gt_noise,gt_noise,ncc_effect)

print(mse/test_count,mse_approx/test_count,mae/test_count,mae_approx/test_count,ssim/test_count,ssim_approx/test_count,nce/test_count )

# %%
