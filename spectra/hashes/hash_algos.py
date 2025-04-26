import numpy as np
import scipy.fftpack


def generate_phash(tensor, height: int, width: int):
    DCT_DIM = 8
    arr = tensor.detach().cpu().numpy().astype(np.uint8).reshape((height, width))
    dct = scipy.fftpack.dct(scipy.fftpack.dct(arr, axis=0), axis=1)
    dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    avg = np.median(dct_kernel)
    diff = dct_kernel > avg
    bitstring = ''.join(['1' if b else '0' for b in diff.flatten()])
    return int(bitstring, 2)