import numpy as np
import scipy.fftpack


def generate_phash(tensor):
    DCT_DIM = 8
    dct = scipy.fftpack.dct(scipy.fftpack.dct(tensor, axis=0), axis=1)
    dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    avg = np.median(dct_kernel)
    diff = dct_kernel > avg
    bitstring = ''.join(['1' if b else '0' for b in diff.flatten()])
    return hex(int(bitstring, 2))