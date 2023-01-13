# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet
import pathlib
import numpy as np
import torch.optim as optim
from fastmri.data import  mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from torch.utils.data import Dataset
import h5py
import math


# %% data loader
from my_data import *

nc = 16
nx = 384
ny = 396

def data_transform(kspace_noisy, kspace_clean, ncc_effect, sense_maps):
    # Transform the kspace to tensor format
    #ncc_effect = transforms.to_tensor(ncc_effect)
    kspace_noisy = transforms.to_tensor(kspace_noisy)
    kspace_noisy = torch.cat((kspace_noisy[torch.arange(nc),:,:].unsqueeze(-1),kspace_noisy[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    #kspace_clean = transforms.to_tensor(kspace_clean)
    #kspace_clean = torch.cat((kspace_clean[torch.arange(nc),:,:].unsqueeze(-1),kspace_clean[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    sense_maps = transforms.to_tensor(sense_maps)
    sense_maps = torch.cat((sense_maps[torch.arange(nc),:,:].unsqueeze(-1),sense_maps[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)

    return kspace_noisy, sense_maps

train_data = SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_copy/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_copy/train/'),
    transform=data_transform,
    challenge='multicoil'
)

def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

# %% MoDL loader
from modl_model import *
layers = 5
iters = 10
recon_model = MoDL(
    n_layers = layers,
    k_iters = iters
)
recon_model = torch.load("/project/jhaldar_118/jiayangw/refnoise/model/modl_mae_acc4_epochs100")

# %% training settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size)
recon_model.to(device)
recon_optimizer = optim.Adam(recon_model.parameters(),lr=1e-3)
L2Loss = torch.nn.MSELoss()
L1Loss = torch.nn.L1Loss()

# %% sampling mask
mask = torch.zeros(ny,dtype=torch.int8)
mask[torch.arange(99)*4] = 1
mask[torch.arange(186,210)] =1
mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(nc,nx,1,2)

# %%
max_epochs = 50
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    batch_count = 0    
    for kspace, sense_maps in train_dataloader:
        batch_count = batch_count + 1
        Mask = mask.unsqueeze(0).repeat(kspace.size(0),1,1,1,1)
        gt = toIm(kspace)

        image_zf = fastmri.complex_mul(fastmri.complex_conj(sense_maps),fastmri.ifft2c(torch.mul(Mask,kspace))) 
        image_zf = torch.permute(torch.sum(image_zf, dim=1),(0,3,1,2)).to(device)
        sense_maps = torch.complex(sense_maps[:,:,:,:,0],sense_maps[:,:,:,:,1]).to(device)
        recon = recon_model(image_zf, sense_maps, Mask[:,0,:,:,0].squeeze().to(device)).to(device)
        recon = fastmri.complex_abs(torch.permute(recon,(0,2,3,1)))

        loss = L1Loss(recon.to(device),gt.to(device))

        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item())
        
        loss.backward()
        recon_optimizer.step()
        recon_optimizer.zero_grad()

    if (epoch + 1)%10 == 0:
        torch.save(recon_model,"/project/jhaldar_118/jiayangw/refnoise/model/modl_mae_acc4_epochs"+str(epoch+101))

# %%
