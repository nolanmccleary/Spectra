from spectra import Attack_Engine, PHASH
from validation import Image_Validator


def phash_attack():
    engine = Attack_Engine()
    validator = Image_Validator("cpu")
    images = [('sample_images/peppers.png', 'output/peppers_attacked.png'), ('sample_images/imagehash.png', 'output/imagehash_attacked.png')]
    engine.add_attack(images, PHASH, 15, 200, "cpu", verbose="on")
    engine.run_attacks()

    for image_pair in images:
        print(validator.compare(image_pair, PHASH))












if __name__ == '__main__':
    phash_attack()