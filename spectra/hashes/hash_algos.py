import math
import numpy as np
import scipy.fftpack
import torch
from spectra.utils import rgb_to_grayscale
from .dsp import create_dct_matrix, jarosz_filter



###################################### AHASH #########################################

def generate_ahash_batched(batched_tensor):
    return torch.stack([_generate_ahash(v) for v in batched_tensor], dim=0)


def generate_ahash_rgb_batched(batched_tensor):
    if batched_tensor.dim() == 3:
        batched_tensor = batched_tensor.unsqueeze(0)
    return torch.stack([_generate_ahash_rgb(v) for v in batched_tensor], dim=0)


def _generate_ahash(tensor):
    mean_tensor = torch.mean(tensor)
    diff = tensor > mean_tensor
    return diff.to(torch.bool).view(-1)



def _generate_ahash_rgb(tensor):
    gray = rgb_to_grayscale(tensor)
    mean_tensor = torch.mean(gray)
    diff = gray > mean_tensor
    return diff.to(torch.bool).view(-1)


###################################### DHASH ##########################################

def generate_dhash_batched(batched_tensor):
    return torch.stack([_generate_dhash(v) for v in batched_tensor], dim=0)


def generate_dhash_rgb_batched(batched_tensor):
    if batched_tensor.dim() == 3:
        batched_tensor = batched_tensor.unsqueeze(0)
    return torch.stack([_generate_dhash_rgb(v) for v in batched_tensor], dim=0)


def _generate_dhash(tensor):
    diff = tensor[:, 1:] > tensor[:, :-1]
    return diff.to(torch.bool).view(-1)


def _generate_dhash_rgb(tensor):
    gray = rgb_to_grayscale(tensor)
    diff = gray[:, 1:] > gray[:, :-1]
    return diff.to(torch.bool).view(-1)



###################################### PHASH ##########################################

def generate_phash_batched(batched_tensor):
    return torch.stack([_generate_phash(v) for v in batched_tensor], dim=0)


def generate_phash_rgb_batched(batched_tensor):
    if batched_tensor.dim() == 3:
        batched_tensor = batched_tensor.unsqueeze(0)
    return torch.stack([_generate_phash_rgb(v) for v in batched_tensor], dim=0)


def _generate_phash(tensor): #[1, H, W] -> [64]
    DCT_DIM = 8
    
    view = tensor.squeeze(0)
    arr = (view.detach().cpu().numpy() * 255).round().astype(np.uint8)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(arr, axis=0), axis=1)
    dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    
    bits = (dct_kernel > np.median(dct_kernel)).astype(np.uint8).flatten()
    return torch.from_numpy(bits).to(tensor.device).to(torch.bool)


def _generate_phash_rgb(tensor): #[1, H, W] -> [64]
    gray = rgb_to_grayscale(tensor)

    DCT_DIM = 8
    view = gray.squeeze(0)
    arr = (view.detach().cpu().numpy() * 255).round().astype(np.uint8)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(arr, axis=0), axis=1)
    dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    
    bits = (dct_kernel > np.median(dct_kernel)).astype(np.uint8).flatten()
    return torch.from_numpy(bits).to(tensor.device).to(torch.bool)



def generate_phash_torch_batched(batched_tensor, dct_dim=8):
    return torch.stack([_generate_phash_torch(v, dct_dim) for v in batched_tensor], dim=0)



#[C, H, W] -> [dct_dim * dct_dim]
def _generate_phash_torch(tensor, dct_dim):
    arr = tensor.squeeze(0)

    H, W = arr.shape
    device, dtype = arr.device, arr.dtype

    D_H = create_dct_matrix(H, device, dtype)[:dct_dim, : ]   # [dim, H]
    D_W = create_dct_matrix(W, device, dtype)[:dct_dim, : ]   # [dim, W]

    #[K,H] @ [H,W] @ [W,K] → [K,K]
    low = D_H @ arr @ D_W.t()

    med = low.median()
    bits = (low > med).flatten()

    return bits.to(torch.bool)



###################################### PDQ ##########################################


def generate_pdq_batched(batched_tensor, dct_dim=16):
    return torch.stack([_generate_pdq(v, dct_dim) for v in batched_tensor], dim=0)



#[C, H, W] -> [dct_dim * dct_dim]
def _generate_pdq(tensor, dct_dim):
    arr = jarosz_filter(tensor).squeeze(0)

    H, W = arr.shape
    device, dtype = arr.device, arr.dtype

    D_H = create_dct_matrix(H, device, dtype)[:dct_dim, : ]   # [dim, H]
    D_W = create_dct_matrix(W, device, dtype)[:dct_dim, : ]   # [dim, W]

    #[K,H] @ [H,W] @ [W,K] → [K,K]
    low = D_H @ arr @ D_W.t()

    med = low.median()
    bits = (low > med).flatten()

    return bits.to(torch.bool)