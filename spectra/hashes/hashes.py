from .hash_wrapper import Hash_Wrapper
from .hash_algos import generate_phash

PHASH = Hash_Wrapper(name="phash", func=generate_phash, resize_height=32, resize_width=32, available_devices={"cpu"})