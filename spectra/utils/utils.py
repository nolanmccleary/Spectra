import torch
import torch.nn.functional as F
import numpy as np



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



def generate_perturbation_vectors_1d(num_perturbations, size, device):
    base = torch.randn((num_perturbations // 2, size), dtype=torch.float32, device=device) #randn implicitly clamps from 0 to 1
    perturbations = torch.cat([base, -base], dim=0)
    return perturbations




def l2_per_pixel_rgb(img1, img2):
    C, H, W = img1.shape
    diff = (img1 - img2).view(C, -1) #Flatten each color's matrix to 1-d array
    return torch.norm(diff, p=2, dim=0).mean().item() #Convert 3xN arrays to 3N array



def hamming_distance_hex(hash1, hash2):
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    xor = int1 ^ int2
    return bin(xor).count('1')



def bit_tensor_sum(packed_tensor):
    count = packed_tensor
    count = (count - ((count >> 1) & 0x5555555555555555))
    count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333)
    count = (count + (count >> 4)) & 0x0F0F0F0F0F0F0F0F
    count = (count * 0x0101010101010101) >> 56
    return torch.sum(count).item()