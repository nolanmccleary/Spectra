import math
import numpy as np
import scipy.fftpack
import torch
from spectra.utils import rgb_to_grayscale


PI = math.pi
ROOT_2 = math.sqrt(2)

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


_dct_cache = {}

def _create_dct_matrix(N, device, dtype): #1) May be able to drop transpose, 2) May be able to drop global scaling

    key = (N, device, dtype)
    if key not in _dct_cache:

        n = torch.arange(N, device=device, dtype=dtype)
        k = n.unsqueeze(0) #[1, N]

        basis = torch.cos(PI * (2 * n + 1).unsqueeze(1) * k / (2 * N)) #[N, 1] * [1, N] -> [N, N]; broadcast across k so we have N dct row vectors of length N
        basis = basis.t().to()

        _dct_cache[key] = basis

    return _dct_cache[key]


#[C, H, W] -> [dct_dim * dct_dim]
def _generate_phash_torch(tensor, dct_dim=8):

    arr = tensor.squeeze(0)

    H, W = arr.shape
    device, dtype = arr.device, arr.dtype

    # get only the top-dct_dim rows of the DCT basis for each axis
    D_H = _create_dct_matrix(H, device, dtype)[:dct_dim, : ]   # [8, H] - Assuming dct_dim = 8; In our case, H should equal W
    D_W = _create_dct_matrix(W, device, dtype)[:dct_dim, : ]   # [8, W]

    # compute low-frequency DCT block: [K,H] @ [H,W] @ [W,K] → [K,K]
    low = D_H @ arr @ D_W.t()

    med = low.median()
    bits = (low > med).flatten()

    return bits.to(torch.bool)



