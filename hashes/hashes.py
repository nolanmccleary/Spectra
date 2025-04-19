from .hash_wrapper import Hash_Wrapper
import hash_algos

PHASH = Hash_Wrapper(name="phash", func=hash_algos.generate_phash(), resize_height=32, resize_width=32, available_devices={"cpu"})