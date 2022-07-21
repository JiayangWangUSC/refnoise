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

# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    image = fastmri.ifft2c(kspace)
    image = image[:,torch.arange(191,575),:,:]
    kspace = fastmri.fft2c(image)/5e-5
    return kspace

train_data = mri_data.SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/test_brain/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain/train/'),
    transform=data_transform,
    challenge='multicoil'
)


def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

# %%
model_pre = torch.load('/project/jhaldar_118/jiayangw/OptSamp/model/uni_model_sigma1')

# %% unet loader
recon_model = Unet(
  in_chans = 32,
  out_chans = 32,
  chans = 128,
  num_pool_layers = 4,
  drop_prob = 0.0
)


# %% training settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
recon_model.to(device)
recon_optimizer = optim.Adam(recon_model.parameters(),lr=3e-4)
L2Loss = torch.nn.MSELoss()

# %% sampling mask
mask = torch.zeros(396)
mask[torch.arange(130)*3] = 1
mask[torch.arange(186,210)] =1
# %%
max_epochs = 50
sigma = 0.4
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    batch_count = 0    
    for train_batch in train_dataloader:
        batch_count = batch_count + 1
        Mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(train_batch.size(0),16,384,1,2).to(device)

    # preprocessing    
        with torch.no_grad():
            image = fastmri.ifft2c(train_batch).to(device)
            image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1).to(device) 
            image_output = model_pre(image_input).to(device)
            image_pre = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4).to(device)
            kspace = fastmri.fft2c(image_pre).to(device)
            gt = toIm(kspace)
            kspace_input = torch.mul(kspace,Mask).to(device)
        
        torch.manual_seed(batch_count)
        noise = torch.mul(sigma*torch.rand_like(kspace_input),Mask)
        image = fastmri.ifft2c(kspace_input).to(device)
        image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1).to(device) 
        image_output = recon_model(image_input).to(device)
        image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4).to(device)
        recon = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)
        
        loss = L2Loss(recon.to(device),gt.to(device))

        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item())
        
        loss.backward()
        recon_optimizer.step()
        recon_optimizer.zero_grad()

    torch.save(recon_model,"/project/jhaldar_118/jiayangw/refnoise/model/model_l2_noise"+str(sigma))
