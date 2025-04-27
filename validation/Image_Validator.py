import lpips
from PIL import Image
from spectra.utils import get_rgb_tensor, lpips_per_pixel_rgb, rgb_to_grayscale, grayscale_resize_and_flatten, hamming_distance_hex
from spectra.hashes import Hash_Wrapper


class Image_Validator:
    

    def __init__(self, device):
        self.device = device



    def compare(self, image_pair: tuple[str], hash: Hash_Wrapper):
        func, resize_height, resize_width, available_devices = hash.get_info()
        
        if self.device not in available_devices:
            raise ValueError(f"Invalid device '{self.device}'. Expected one of: {available_devices}")
        
        input_image = None
        output_image = None

        with Image.open(image_pair[0]) as img:
            input_image = get_rgb_tensor(img, self.device)
        
        with Image.open(image_pair[1]) as img:
            output_image = get_rgb_tensor(img, self.device)
        
        lpips_per_pixel = lpips_per_pixel_rgb(input_image, output_image, lpips.LPIPS(net='alex').to(self.device))
        
        
        grayscale = rgb_to_grayscale(input_image) #(1, H, W)
        grayscale_resized = grayscale_resize_and_flatten(grayscale, resize_height, resize_width) #[HxW]
        original_hash = func(grayscale_resized, resize_height, resize_width)

        grayscale = rgb_to_grayscale(output_image) #(1, H, W)
        grayscale_resized = grayscale_resize_and_flatten(grayscale, resize_height, resize_width)
        output_hash = func(grayscale_resized, resize_height, resize_width)

        hamming_distance = hamming_distance_hex(original_hash, output_hash)

        return {
            "original_hash" : hex(original_hash),
            "output_hash": hex(output_hash),
            "hamming_distance": hamming_distance,
            "lpips_per_pixel": lpips_per_pixel
        }