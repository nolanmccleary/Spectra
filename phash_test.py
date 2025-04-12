from PIL import Image
import imagehash
import numpy as np
import random




NUM_VECTORS = 800
IMAGE_STRING = 'sample_images/peppers.png'

DCT_DIM = 8
DCT_HFF = 4


MAX_CYCLES = 50000




def process_image(image, image_size):
    image = image.convert('L').resize((image_size, image_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image)
    return pixels


def generate_perturbation_vectors(image_arr):
    dim = image_arr.size
    half = NUM_VECTORS // 2
    base = np.random.randn(half, dim) #Create <half> vectors of length <dim>
    base_normalized = (base - np.min(base)) / (np.max(base) - np.min(base))
    perturbations = np.vstack([base_normalized, -base_normalized])
    return perturbations





def image_pipeline():
    print('start')
    
    size = DCT_DIM * DCT_HFF
    image = process_image(Image.open(IMAGE_STRING), size)
    perturbations = generate_perturbation_vectors(image)
    
    for _ in range(MAX_CYCLES):
        i = random.randint(0, NUM_VECTORS-1)
        print(perturbations[i])
    


    print('end')




image_pipeline()




