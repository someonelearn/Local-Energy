import torch
from PIL import Image
from noise_addition import GaussianNoise

def cal_energy_score(latent):
    #x.shape = (bsize, score)
    y_pred = latent[0].argmax(dim=-1, keepdim=True)
    Energy = -latent[:, y_pred]
    return torch.mean(Energy, dim=0, keepdim=True)
