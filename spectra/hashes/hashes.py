from .hash_wrapper import Hash_Wrapper
from .hash_algos import generate_phash, generate_phash_rgb

PHASH = Hash_Wrapper(name="phash", func=generate_phash, grayscale=True, resize_height=32, resize_width=32, available_devices={"cpu"})

PHASH_RGB = Hash_Wrapper(name="phash_rgb", func=generate_phash_rgb, grayscale=False, resize_height=32, resize_width=32, available_devices={"cpu"})