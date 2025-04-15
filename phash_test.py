#TODO: Allow for non-square image input; absorb full design thus far
#TODO: Plot hash optimization surface
#TODO: Plot performance metrics WRT NUM_PERTURBATIONS and MAX_CYCLES; 

#TODO: Add seed, inverse delta
#TODO: CUDA Parallelism -> CUDA DCT -> CUDA PHASH -> CUDA Hashgrad
#TODO: LPIPS


from PIL import Image
#import imagehash
import numpy as np
import scipy
import math


'''
def process_image(image, image_size):
    image = image.convert('L').resize((image_size, image_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype='float64')
    return pixels.flatten()



def generate_perturbation_vectors(image_arr, half_size):
    dim = image_arr.size
    base = np.random.randn(half_size, dim)
    perturbations = np.vstack([base, -base])
    return perturbations



def generate_phash(pixels, height, width, dct_dim):
    pixels_2d = pixels.reshape((height, width))    
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels_2d, axis=0), axis=1)
    dct_kernel = dct[1 : dct_dim + 1, 1 : dct_dim + 1] #Choose this indexing as DC term contribution to image structure is minimal
    #dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    avg = np.median(dct_kernel)
    diff = dct_kernel > avg
    bitstring = ''.join(['1' if b else '0' for b in diff.flatten()])
    return hex(int(bitstring, 2))



def l2_per_pixel(img1, img2, imgsize):
    delta_arr = (img1 - img2) ** 2
    sum = 0
    for element in delta_arr:
        sum += element
    sum = math.sqrt(sum)
    return sum / imgsize



def hamming_distance_hex(hex1, hex2):
    int1 = int(hex1, 16)
    int2 = int(hex2, 16)
    xor_result = int1 ^ int2
    return bin(xor_result).count('1')



def phash_attack():
    NUM_PERTURBATIONS = 3000 
    IMAGE_STRING = 'sample_images/peppers.png'
    DCT_DIM = 8
    DCT_HFF = 4
    DCT_SIDE_LENGTH = DCT_DIM * DCT_HFF
    SCALE_FACTOR = 6                       #Higher: l2 rises quicker but solution found faster; lower, l2 rises slower but solution takes longer to discover or not discovered at all (this could result in an overall larger l2 distance as l2 increases monotonically as the program runs (statistically speaking))
    STEP_SIZE = 0.01
    MAX_CYCLES = 10000
    HASH_THRESHOLD = 8
    
    rgb_image = Image.open(IMAGE_STRING)

    converted_image = process_image(rgb_image, DCT_SIDE_LENGTH)
    converted_image_size = converted_image.size

    initial_hash = generate_phash(converted_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)

    perturbations = generate_perturbation_vectors(converted_image, NUM_PERTURBATIONS // 2)
    current_image = converted_image.copy()  #Numpy by passes arrays by reference; I don't even wanna say how much time I spent trying to debug this.
    total_delta = np.zeros(converted_image_size)

    nruns = 0

    for _ in range(MAX_CYCLES):
        nruns += 1
        
        gradient = np.zeros(converted_image_size, dtype='float64')

        #TODO: Parallelize this
        for i in range(NUM_PERTURBATIONS):
            scaled_perturbation = perturbations[i] * SCALE_FACTOR
            hash_delta = hamming_distance_hex(generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM), generate_phash(current_image + scaled_perturbation, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)) #discrete hamming delta lowers gradient sensitiivity
            gradient += hash_delta * scaled_perturbation #could possibly scale this
                
        delta = np.sign(gradient) * STEP_SIZE
        current_image += delta
        total_delta += delta

        ham = hamming_distance_hex(generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM), initial_hash)
        l2 = l2_per_pixel(current_image, converted_image, converted_image_size)
        print("HAM: " + str(ham) + " L2: " + str(l2))

        if ham >= HASH_THRESHOLD:
            break

    print(initial_hash)
    current_hash = generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
    print(current_hash)
    print(hamming_distance_hex(initial_hash, current_hash))
    print(l2_per_pixel(current_image, converted_image, converted_image_size))
    print(nruns)

'''


import torch
from PIL import Image
import numpy as np
import scipy.fftpack
import math
from torchvision import transforms


def l2_per_pixel(img1, img2, imgsize):
    delta_arr = (img1 - img2) ** 2
    sum = 0
    for element in delta_arr:
        sum += element
    sum = math.sqrt(sum)
    return sum / imgsize



# Use Metal on Mac, CUDA on Linux/Windows, fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")


def process_image(image, image_size):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    tensor = transform(image).squeeze(0).to(torch.float32)  # Shape: (H, W)
    return tensor.flatten().to(device)


def generate_perturbation_vectors(dim, half_size):
    base = torch.randn((half_size, dim), dtype=torch.float32, device=device)
    perturbations = torch.cat([base, -base], dim=0)
    return perturbations


def generate_phash(tensor, height, width, dct_dim):
    pixels_2d = tensor.reshape((height, width)).cpu().numpy()
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels_2d, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_kernel = dct[1:dct_dim+1, 1:dct_dim+1]
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
    IMAGE_PATH = 'sample_images/peppers.png'
    DCT_DIM = 8
    DCT_HFF = 4
    DCT_SIDE_LENGTH = DCT_DIM * DCT_HFF
    SCALE_FACTOR = 6
    STEP_SIZE = 0.01
    MAX_CYCLES = 10000
    HASH_THRESHOLD = 34

    rgb_image = Image.open(IMAGE_PATH)
    converted_image = process_image(rgb_image, DCT_SIDE_LENGTH)
    converted_image_size = converted_image.numel()

    initial_hash = generate_phash(converted_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)

    perturbations = generate_perturbation_vectors(converted_image_size, NUM_PERTURBATIONS // 2)
    current_image = converted_image.clone()
    total_delta = torch.zeros_like(current_image)

    for cycle in range(MAX_CYCLES):
        gradient = torch.zeros_like(current_image)

        # Compute gradient estimate
        for i in range(NUM_PERTURBATIONS):
            scaled_pert = perturbations[i] * SCALE_FACTOR
            h1 = generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
            h2 = generate_phash(current_image + scaled_pert, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
            delta_ham = hamming_distance_hex(h1, h2)
            gradient += delta_ham * scaled_pert

        delta = torch.sign(gradient) * STEP_SIZE
        current_image += delta
        total_delta += delta

        ham = hamming_distance_hex(generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM), initial_hash)
        l2 = l2_per_pixel(current_image, converted_image, converted_image_size)

        print(f"HAM: {ham} L2: {l2:.4f}")

        if ham >= HASH_THRESHOLD:
            break

    final_hash = generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
    print("\nInitial Hash:", initial_hash)
    print("Final Hash:  ", final_hash)
    print("Final HAM:   ", hamming_distance_hex(initial_hash, final_hash))
    print("Final L2 torch:    ", l2_per_pixel_torch(current_image, converted_image, converted_image_size))
    print("Final L2 nontorch:    ", l2_per_pixel(current_image, converted_image, converted_image_size))
    print("Cycles Run:  ", cycle + 1)

phash_attack()














#phash_attack()




