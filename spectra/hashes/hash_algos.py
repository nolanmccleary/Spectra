import numpy as np
import scipy.fftpack
import torch
from spectra.utils import rgb_to_grayscale, rgb_to_luma


def generate_phash(tensor): #[1, H, W] -> [64]
    DCT_DIM = 8
    view = tensor.squeeze(0)
    arr = (view.detach().cpu().numpy() * 255).round().astype(np.uint8)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(arr, axis=0), axis=1)
    dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    
    bits = (dct_kernel > np.median(dct_kernel)).astype(np.uint8).flatten()
    return torch.from_numpy(bits).to(tensor.device).to(torch.bool)


def generate_phash_rgb(tensor): #[1, H, W] -> [64]
    gray = rgb_to_grayscale(tensor)
    
    DCT_DIM = 8
    view = gray.squeeze(0)
    arr = (view.detach().cpu().numpy() * 255).round().astype(np.uint8)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(arr, axis=0), axis=1)
    dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    
    bits = (dct_kernel > np.median(dct_kernel)).astype(np.uint8).flatten()
    return torch.from_numpy(bits).to(tensor.device).to(torch.bool)
