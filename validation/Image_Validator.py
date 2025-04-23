from PIL import Image
from spectra.utils import get_rgb_tensor, l2_per_pixel_rgb, rgb_to_grayscale, grayscale_resize_and_flatten, hamming_distance_hex
from spectra.hashes import Hash_Wrapper


class Image_Validator:
    

    def __init__(self, device):
        self.device = device
        self.input_image = None
        self.output_image = None
        self.original_hash = None
        self.output_hash = None
        self.output_hamming = 0
        self.l2_per_pixel = 0



    def compare(self, image_pair: tuple[str], hash: Hash_Wrapper):
        self.set_image_pair(image_pair)
        self.get_l2_per_pixel()
        self.get_hashes(hash)
        self.get_hamming_distance()
        return {
            "original_hash" : hex(self.original_hash),
            "output_hash": hex(self.output_hash),
            "hamming_distance": self.output_hamming,
            "l2_per_pixel": self.l2_per_pixel
        }



    def set_image_pair(self, image_pair: tuple[str]):
        with Image.open(image_pair[0]) as img:
            self.input_image = get_rgb_tensor(img, self.device)

        with Image.open(image_pair[1]) as img:
            self.input_image = get_rgb_tensor(img, self.device)



    def get_hashes(self, hash: Hash_Wrapper):
        name, func, resize_height, resize_width, available_devices = hash.get_info()
        if self.device not in available_devices:
            raise ValueError(f"Invalid device '{self.device}'. Expected one of: {available_devices}")
        
        grayscale = rgb_to_grayscale(self.input_image) #(1, H, W)
        grayscale_resized = grayscale_resize_and_flatten(grayscale, resize_height, resize_width) #[HxW]
        self.original_hash = func(grayscale_resized, resize_height, resize_width)

        grayscale = rgb_to_grayscale(self.output_image) #(1, H, W)
        grayscale_resized = grayscale_resize_and_flatten(grayscale, resize_height, resize_width)
        self.output_hash = func(grayscale_resized, resize_height, resize_width)



    def get_hamming_distance(self):
        return hamming_distance_hex(self.original_hash, self.output_hash)



    def get_l2_per_pixel(self):
        self.l2_per_pixel = l2_per_pixel_rgb(self.input_image, self.output_image)