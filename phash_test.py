#TODO: Allow for non-square image input


#ISSUE: Small step distances result in hash gradient not being computeable

from PIL import Image
import imagehash
import numpy as np
import random
import scipy



NUM_PERTURBATIONS= 200
IMAGE_STRING = 'sample_images/peppers.png'

DCT_DIM = 16
DCT_HFF = 4
DCT_SIDE_LENGTH = DCT_DIM * DCT_HFF

SCALE_FACTOR = 2
STEP_SIZE = 0.01


MAX_CYCLES = 100



def process_image(image, image_size):
    image = image.convert('L').resize((image_size, image_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype='float64')
    return pixels.flatten()



def generate_perturbation_vectors(image_arr):
    dim = image_arr.size
    half = NUM_PERTURBATIONS // 2
    base = np.random.randn(half, dim) #Create <half> vectors of length <dim>
    base_normalized = (base - np.min(base, axis=1, keepdims=True)) / (np.max(base, axis=1, keepdims=True) - np.min(base, axis=1, keepdims=True))
    perturbations = np.vstack([base_normalized, -base_normalized])
    return perturbations



def generate_phash(pixels):
    height = DCT_SIDE_LENGTH
    width = DCT_SIDE_LENGTH
    pixels_2d = pixels.reshape((height, width))    
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels_2d, axis=0), axis=1)
    #dct_kernel = dct[1 : DCT_DIM + 1, 1 : DCT_DIM + 1] #Choose this indexing as DC term contribution to image structure is minimal
    dct_kernel = dct[:DCT_DIM, :DCT_DIM]
    avg = np.median(dct_kernel)
    diff = dct_kernel > avg
    bitstring = ''.join(['1' if b else '0' for b in diff.flatten()])
    return hex(int(bitstring, 2))



def hamming_distance_hex(hex1, hex2):
    int1 = int(hex1, 16)
    int2 = int(hex2, 16)
    xor_result = int1 ^ int2
    return bin(xor_result).count('1')



def perturbation_hash_delta(image, perturbation):
    new_image = image + perturbation
    return hamming_distance_hex(generate_phash(image), generate_phash(new_image))



def image_pipeline():

    rgb_image = Image.open(IMAGE_STRING)

    converted_image = process_image(rgb_image, DCT_SIDE_LENGTH)
    converted_image_size = converted_image.size

    initial_hash = generate_phash(converted_image)

    

    perturbations = generate_perturbation_vectors(converted_image)
    current_image = converted_image.copy()
    total_delta = np.zeros(converted_image_size)

    for _ in range(MAX_CYCLES):
        gradient = np.zeros(converted_image_size, dtype='float64')

        #TODO: Parallelize this
        for i in range(NUM_PERTURBATIONS):
            scaled_perturbation = perturbations[i] * SCALE_FACTOR
            hash_delta = perturbation_hash_delta(current_image, scaled_perturbation) #discrete hamming delta lowers gradient sensitiivity
            gradient += hash_delta * scaled_perturbation
                
        delta = np.sign(gradient) * STEP_SIZE

        current_image += delta
        total_delta += delta
        print(total_delta)



    #for i in range(current_image.size):
    #    if current_image[i] != converted_image[i]:
    #        print(converted_image[i])
    #        print(current_image[i])
            

    print(initial_hash)
    print(generate_phash(current_image))

image_pipeline()




