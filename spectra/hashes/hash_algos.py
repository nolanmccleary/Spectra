import numpy as np
import scipy.fftpack
import torch


def generate_phash(tensor, height, width):
    DCT_DIM = 8
    arr = (tensor.detach().cpu().numpy() * 255).round().astype(np.uint8).reshape((height, width))
    dct = scipy.fftpack.dct(scipy.fftpack.dct(arr, axis=0), axis=1)
    dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    
    bits = (dct_kernel > np.median(dct_kernel)).astype(np.uint8).flatten()
    return torch.from_numpy(bits).to(tensor.device).to(torch.bool)
    
    
    '''
    avg = np.median(dct_kernel)
    diff = dct_kernel > avg
    bitstring = ''.join(['1' if b else '0' for b in diff.flatten()])
    return int(bitstring, 2)
    '''