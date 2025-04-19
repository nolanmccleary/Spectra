import torch
import torch.nn.functional as F
import numpy as np
import scipy.fftpack



def get_rgb_tensor(image_object, rgb_device):
    if image_object.mode == 'RGBA':
        image_object = image_object.convert('RGB')
    arr = np.array(image_object).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(rgb_device)
    return tensor



def rgb_to_grayscale(rgb_tensor):
    gray = rgb_tensor.mean(dim=0, keepdim=True)    # [1, H, W]
    return gray



def grayscale_resize(grayscale_tensor, height, width):
    grayscale_tensor = grayscale_tensor.clone().unsqueeze(0)
    gray_resized = F.interpolate(
        grayscale_tensor,
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )
    return gray_resized.view(-1)



def inverse_delta(rgb_tensor, delta, eps=1e-6):
    C, H, W = rgb_tensor.shape

    if delta.shape == (C, H, W):
        return delta

    rgb_flat = rgb_tensor.view(C, -1)
    rgb_mean = rgb_flat.mean(dim=0, keepdim=True)  # [1, H*W]
    gd = delta.unsqueeze(0)              # [1, H*W]
    # Avoid division by zero
    delta = torch.where(
        gd <= 0,
        gd * rgb_flat / (rgb_mean + eps),
        gd * (1 - rgb_flat) / ((1 - rgb_mean) + eps)
    )

    return delta.view(C, H, W)



def generate_seed_perturbation(dim, start_scalar, device):
    return torch.rand((1, dim), dtype=torch.float32, device=device) * start_scalar



def generate_perturbation_vectors_1d(num_perturbations, half_size, device):
    base = torch.randn((half_size, num_perturbations), dtype=torch.float32, device=device) #randn implicitly clamps from 0 to 1
    perturbations = torch.cat([base, -base], dim=0)
    return perturbations



def generate_phash(tensor, height, width, dct_dim):
    pixels_2d = tensor.reshape((height, width)).cpu().numpy()
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels_2d, axis=0), axis=1)
    dct_kernel = dct[:dct_dim, :dct_dim]
    avg = np.median(dct_kernel)
    diff = dct_kernel > avg
    bitstring = ''.join(['1' if b else '0' for b in diff.flatten()])
    return hex(int(bitstring, 2))



def l2_per_pixel_rgb(img1, img2):
    C, H, W = img1.shape
    diff = (img1 - img2).view(C, -1) #Flatten each color's matrix to 1-d array
    return torch.norm(diff, p=2, dim=0).mean().item() #Convert 3xN arrays to 3N array



def hamming_distance_hex(hash1, hash2):
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    xor = int1 ^ int2
    return bin(xor).count('1')



