#TODO: Add seed, inverse delta ; Inverse delta done, still need to test
#TODO: CUDA Parallelism -> CUDA DCT -> CUDA PHASH -> CUDA Hashgrad ; CUDA Parallelism done, move to DCT.
#TODO: LPIPS instead of L2
#TODO: Need to implement some sort of perturbation scaling algo


import torch
import torch.nn.functional as Func
from PIL import Image
import numpy as np
import scipy.fftpack
from torchvision.transforms import ToPILImage


device = torch.device("cpu")


def get_rgb_tensor(image_object, rgb_device):
    if image_object.mode == 'RGBA':
        image_object = image_object.convert('RGB')
    arr = np.array(image_object).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(rgb_device)
    return tensor



def get_grayscale_tensor(rgb_tensor, grayscale_side_len, grayscale_device):
    C, H, W = rgb_tensor.shape
    # Mean over channels
    gray = rgb_tensor.mean(dim=0, keepdim=True)    # [1, H, W]
    gray = gray.unsqueeze(0)                       # [1, 1, H, W]
    gray_resized = Func.interpolate(
        gray,
        size=(grayscale_side_len, grayscale_side_len),
        mode='bilinear',
        align_corners=False
    )
    return gray_resized.view(-1).to(grayscale_device)



def inverse_delta(rgb_tensor, grayscale_delta, eps=1e-6):
    C, H, W = rgb_tensor.shape
    rgb_flat = rgb_tensor.view(C, -1)
    rgb_mean = rgb_flat.mean(dim=0, keepdim=True)  # [1, H*W]
    gd = grayscale_delta.unsqueeze(0)              # [1, H*W]
    # Avoid division by zero
    delta = torch.where(
        gd <= 0,
        gd * rgb_flat / (rgb_mean + eps),
        gd * (1 - rgb_flat) / ((1 - rgb_mean) + eps)
    )
    return delta.view(C, H, W)



def generate_perturbation_vectors(dim, half_size):
    base = torch.randn((half_size, dim), dtype=torch.float32, device=device)
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



def l2_per_pixel_torch(img1, img2):
    C, H, W = img1.shape
    diff = (img1 - img2).view(C, -1)
    return torch.norm(diff, p=2, dim=0).mean().item()



def hamming_distance_hex(hash1, hash2):
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    xor = int1 ^ int2
    return bin(xor).count('1')



# Main attack function
def phash_attack():
    NUM_PERTURBATIONS = 3000
    INPUT_IMAGE_PATH = 'sample_images/peppers.png'
    OUTPUT_IMAGE_PATH = 'output/peppers_attacked.png'
    DCT_DIM = 8
    DCT_HFF = 4
    DCT_SIDE_LENGTH = DCT_DIM * DCT_HFF
    SCALE_FACTOR = 6.0
    STEP_SIZE = 0.01
    MAX_CYCLES = 10000
    HASH_THRESHOLD = 10

    rgb_image = Image.open(INPUT_IMAGE_PATH)
    rgb_tensor = get_rgb_tensor(rgb_image, device)
    grayscale_tensor = get_grayscale_tensor(rgb_tensor, DCT_SIDE_LENGTH, device)

    initial_hash = generate_phash(grayscale_tensor, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)

    dim = grayscale_tensor.numel()
    perturbations = generate_perturbation_vectors(dim, NUM_PERTURBATIONS // 2)
    current_image = grayscale_tensor.clone()
    total_delta = torch.zeros_like(current_image)


    for _ in range(MAX_CYCLES):
        gradient = torch.zeros_like(current_image)
        ph_curr = generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
        for i in range(NUM_PERTURBATIONS):
            scaled_pert = perturbations[i] * SCALE_FACTOR
            h2 = generate_phash(current_image + scaled_pert, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
            delta_ham = hamming_distance_hex(ph_curr, h2)
            gradient.add_(delta_ham * scaled_pert)
        delta = gradient.sign() * STEP_SIZE
        current_image = (current_image + delta).clamp(0.0, 1.0)
        total_delta.add_(delta)
        ham = hamming_distance_hex(generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM), initial_hash)
        if ham >= HASH_THRESHOLD:
            break


    final_hash = generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
    final_ham = hamming_distance_hex(initial_hash, final_hash)


    small = total_delta.view(1, 1, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH)
    C, H, W = rgb_tensor.shape
    upsampled_delta = Func.interpolate(
        small,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).view(-1)
    image_delta = inverse_delta(rgb_tensor, upsampled_delta)


    output_tensor = (rgb_tensor + image_delta).clamp(0.0, 1.0)
    final_l2 = l2_per_pixel_torch(rgb_tensor, output_tensor)

    # Print results
    print("Initial Hash:", initial_hash)
    print("Final Hash:  ", final_hash)
    print("Final HAM:   ", final_ham)
    print("Final L2 torch:", final_l2)

    # Save image
    out = output_tensor.detach().cpu()
    output_image = ToPILImage()(out)
    output_image.save(OUTPUT_IMAGE_PATH)
    print(f"Saved attacked image to {OUTPUT_IMAGE_PATH}")



if __name__ == '__main__':
    phash_attack()