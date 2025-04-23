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
    grayscale_tensor = grayscale_tensor.clone().unsqueeze(1)
    gray_resized = F.interpolate(   #Interpolate needs to know batch and channel dimensions thus a 4-d tensor is required
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
    base = torch.randn((num_perturbations // 2, size), dtype=torch.float32, device=device) 
    absmax = base.abs().amax(dim=1, keepdim=True)
    scale = torch.where(absmax > 0, 1.0 / absmax, torch.tensor(1.0, device = device)) #scale tensor to get max val of each generated perturbation so that we can normalize
    base = base * scale
    return torch.cat([base, -base], dim=0) #Mirror to preserve distribution
    



def l2_per_pixel_rgb(img1, img2):
    C, H, W = img1.shape
    diff = (img1 - img2).view(C, -1) #Flatten each color's matrix to 1-d array
    return torch.norm(diff, p=2, dim=0).mean().item() #Convert 3xN arrays to 3N array



def hamming_distance_hex(a, b):
    # turn a into a plain Python int
    if isinstance(a, str):
        ai = int(a, 16)
        print("STRING")
    elif torch.is_tensor(a):
        ai = int(a.item())
        print("TENSOR")
    else:
        ai = int(a)

    # same for b
    if isinstance(b, str):
        bi = int(b, 16)
        print("STRING")
    elif torch.is_tensor(b):
        bi = int(b.item())
        print("TENSOR")
    else:
        bi = int(b)

    # now just XOR and count bits
    return (ai ^ bi).bit_count()


MASK64 = (1 << 64) - 1 #0xFFFFFFFFFFFFFFFF
SIGN_BIT = 1 << 63
OFFSET64 = 1 << 64
def to_signed_int64(u64):
    u64 &= MASK64
    return u64 if u64 < SIGN_BIT else u64 - OFFSET64    #If > 2^63-1 



def popcoint(packed_tensor):
    count = packed_tensor
    count = (count - ((count >> 1) & 0x5555555555555555)) #Paiwise sums; each 2-bit slot now holds b_i + b_{i+1}
    count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333) #Sum pairwise sums into nibbles; each 4-bit slot now has b_i +...+ b_{i+3}
    count = (count + (count >> 4)) & 0x0F0F0F0F0F0F0F0F #Sum top nibbles into bytes; each 8-bit slot now has b_i +...+ b_{i+7}
    count = (count * 0x0101010101010101) >> 56 #Sum byte values into top byte then right-shift to get popc(oin)ount; byte7 = byte0 +...+byte7 = popc(oin)ount << 56
    #print("POPCOINT")
    #print(count)
    return count

