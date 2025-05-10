import os
import sys
import torch
from spectra import Attack_Engine, PHASH, PHASH_RGB
from validation import phash_compare, PDQ_compare


torch.set_default_dtype(torch.float64)


sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))

def phash_attack():
    engine = Attack_Engine(verbose="on")
    #validator = Image_Validator("cpu")
    images = [('sample_images/peppers.png', 'output/peppers_attacked.png'), ('sample_images/peppers.jpeg', 'output/peppers_attacked.jpeg')]
    engine.add_attack(images, PHASH, 20, 100, "cpu", verbose="off")
    engine.run_attacks()

    for image_pair in images:
        #print(validator.compare(image_pair, PHASH))
        print(phash_compare(image_pair[0], image_pair[1]))
        print(PDQ_compare(image_pair[0], image_pair[1]))




if __name__ == '__main__':
    phash_attack()