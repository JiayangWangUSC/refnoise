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

# %% 
def whiten_matrix(noise):
    
    noise = torch.complex(noise[:,:,:,0],noise[:,:,:,1])
    noise = torch.reshape(noise,(16,-1))
    noise = torch.add(noise,-torch.mean(noise,1).unsqueeze(1).repeat(1,noise.size(1)))
    cov = noise@noise.T.conj()/(noise.size(1)-1)
    cov_inv = torch.linalg.inv(cov)
    L = torch.linalg.cholesky(cov_inv)
    return L.T.conj()

# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    # undersample 
    image = fastmri.ifft2c(kspace)
    noise = torch.cat((image[:,torch.arange(191),:,:],image[:,torch.arange(575,768),:,:]),1)
    image = image[:,torch.arange(191,575),:,:]
    # whitening
    W = whiten_matrix(noise)
    image = W @ torch.reshape(torch.complex(image[:,:,:,0],image[:,:,:,1]),(16,-1))
    image = torch.reshape(image,(16,384,396))
    return image

test_data = mri_data.SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/test_brain/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain/train/'),
    transform=data_transform,
    challenge='multicoil'
)


# %%
image = test_data[0]