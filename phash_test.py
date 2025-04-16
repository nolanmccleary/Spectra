#TODO: Add seed, inverse delta
#TODO: CUDA Parallelism -> CUDA DCT -> CUDA PHASH -> CUDA Hashgrad ; CUDA Parallelism done, move to DCT.
#TODO: LPIPS instead of L2


import torch
from PIL import Image
import numpy as np
import scipy.fftpack
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage




device = torch.device("cpu")#"mps" if torch.backends.mps.is_available()
                      #else "cuda" if torch.cuda.is_available()
                      #else "cpu")



def get_rgb_tensor(image_object, rgb_device):
    if image_object.mode == 'RGBA':
        image_object = image_object.convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(image_object).to(torch.float16)
    return tensor.to(rgb_device)



def get_grayscale_tensor(rgb_tensor, grayscale_side_len, grayscale_device):
    tensor = rgb_tensor.clone()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((grayscale_side_len, grayscale_side_len))
    ])
    tensor = transform(tensor).to(torch.float16)
    return tensor.flatten().to(grayscale_device)



'''
def inverse_delta(rgb_tensor, grayscale_delta, grayscale_delta_size):
    C, H, W = rgb_tensor.shape
    rgb_tensor = rgb_tensor.view(C, -1) #Flatten - Ex 3x32x32 -> 3x1024
    rgb_delta = torch.zeros_like(rgb_tensor)
    rgb_mean = (rgb_tensor[0] + rgb_tensor[1] + rgb_tensor[2]) / 3

    for i in range(C):
        for j in range(grayscale_delta_size): #R, G, B
            if grayscale_delta[j] <= 0:
                rgb_delta[i, j] = grayscale_delta[j] * rgb_tensor[i, j] / rgb_mean[j]
            else:
                rgb_delta[i, j] = grayscale_delta[j] * ((1 - rgb_tensor[i, j]) / (1 - rgb_mean[j]))
            rgb_delta[i, j] = min(max(rgb_tensor[i, j] + rgb_delta[i, j], 0), 1) - rgb_tensor[i, j] #Make sure it's within bounds

    return rgb_delta.view(C, H, W) 
'''




def inverse_delta(rgb_tensor, grayscale_delta, grayscale_delta_size):
    # rgb_tensor: [3, H, W]
    C, H, W = rgb_tensor.shape

    rgb_tensor = rgb_tensor.view(C, -1)                       
    assert rgb_tensor.size(1) == grayscale_delta_size, (
        f"expected grayscale_delta_size={rgb_tensor.size(1)}, got {grayscale_delta_size}"
    )

    # per‑pixel mean over R,G,B
    rgb_mean = (rgb_tensor[0] + rgb_tensor[1] + rgb_tensor[2]) / 3  # shape: (H*W,)

    # make both into shape (1, H*W) so they broadcast over the 3 channels
    gd = grayscale_delta.unsqueeze(0)       # [1, H*W]
    rm = rgb_mean.unsqueeze(0)              # [1, H*W]

    rgb_delta = torch.where(
        gd <= 0,
        gd * rgb_tensor     / rm,           
        gd * (1 - rgb_tensor) / (1 - rm)     
    )                                       

    #rgb_delta = (rgb_tensor + rgb_delta).clamp(0., 1.) - rgb_tensor
    return rgb_delta.view(C, H, W)





def generate_perturbation_vectors(dim, half_size):
    base = torch.randn((half_size, dim), dtype=torch.float16, device=device)
    perturbations = torch.cat([base, -base], dim=0)
    return perturbations



def generate_phash(tensor, height, width, dct_dim):
    pixels_2d = tensor.reshape((height, width)).cpu().numpy()               #Reshape to cpu for now as the stock dct doesn't have GPU support
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels_2d, axis=0), axis=1)
    #dct_kernel = dct[1:dct_dim+1, 1:dct_dim+1]
    dct_kernel = dct[:dct_dim, :dct_dim]
    avg = np.median(dct_kernel)
    diff = dct_kernel > avg
    bitstring = ''.join(['1' if b else '0' for b in diff.flatten()])
    return hex(int(bitstring, 2))



def l2_per_pixel_torch(img1, img2, imgsize):
    return torch.linalg.norm(img1 - img2) / imgsize



def hamming_distance_hex(hex1, hex2):
    int1 = int(hex1, 16)
    int2 = int(hex2, 16)
    xor_result = int1 ^ int2
    return bin(xor_result).count('1')



def phash_attack():
    NUM_PERTURBATIONS = 3000
    INPUT_IMAGE_PATH = 'sample_images/peppers.png'
    OUTPUT_IMAGE_PATH = 'output/peppers_attacked.png'
    DCT_DIM = 8
    DCT_HFF = 4
    DCT_SIDE_LENGTH = DCT_DIM * DCT_HFF
    SCALE_FACTOR = 6
    STEP_SIZE = 0.01
    MAX_CYCLES = 10000
    HASH_THRESHOLD = 32

    rgb_image = Image.open(INPUT_IMAGE_PATH)    
    rgb_tensor = get_rgb_tensor(rgb_image, device)
    grayscale_tensor = get_grayscale_tensor(rgb_tensor, DCT_SIDE_LENGTH, device)
    
    grayscale_tensor_size = grayscale_tensor.numel()

    initial_hash = generate_phash(grayscale_tensor, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)

    perturbations = generate_perturbation_vectors(grayscale_tensor_size, NUM_PERTURBATIONS // 2)
    current_image = grayscale_tensor.clone()
    total_delta = torch.zeros_like(current_image)

    for cycle in range(MAX_CYCLES):
        gradient = torch.zeros_like(current_image)

        # Compute gradient estimate; this can be more efficiently parallelized
        for i in range(NUM_PERTURBATIONS):
            scaled_pert = perturbations[i] * SCALE_FACTOR
            h1 = generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
            h2 = generate_phash(current_image + scaled_pert, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
            delta_ham = hamming_distance_hex(h1, h2)
            gradient.add_(delta_ham * scaled_pert)

        delta = torch.sign(gradient) * STEP_SIZE
        current_image.add_(delta)
        total_delta.add_(delta)

        ham = hamming_distance_hex(generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM), initial_hash)
        l2 = l2_per_pixel_torch(current_image, grayscale_tensor, grayscale_tensor_size)

        print(f"HAM: {ham} L2: {l2:.4f}")

        if ham >= HASH_THRESHOLD:
            break

    final_hash = generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
    print("\nInitial Hash:", initial_hash)
    print("Final Hash:  ", final_hash)
    print("Final HAM:   ", hamming_distance_hex(initial_hash, final_hash))
    print("Final L2 torch:    ", l2_per_pixel_torch(current_image, grayscale_tensor, grayscale_tensor_size))
    print("Cycles Run:  ", cycle + 1)

    image_delta = inverse_delta(rgb_tensor, total_delta, grayscale_tensor_size)
    output_tensor = rgb_tensor + image_delta
    out = output_tensor.detach().cpu().clamp(0, 1)
    output_image = ToPILImage(out)
    output_image.save(OUTPUT_IMAGE_PATH)
    

phash_attack()


