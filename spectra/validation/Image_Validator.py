import lpips
from PIL import Image
from spectra.utils import get_rgb_tensor, lpips_rgb
from spectra.hashes import Hash_Wrapper
from .third_party_validation import phash_compare, PDQ_compare




def image_compare(image_pair: tuple[str], device="cpu"):
    
    input_image = None
    output_image = None

    with Image.open(image_pair[0]) as img:
        input_image = get_rgb_tensor(img, device)
    
    with Image.open(image_pair[1]) as img:
        output_image = get_rgb_tensor(img, device)
    
    lpips_score = lpips_rgb(input_image, output_image, lpips.LPIPS(net='alex').to(device))
    
    img1, img2 = image_pair
    phash_delta = phash_compare(img1, img2)["hamming"]
    pdq_delta = PDQ_compare(img1, img2)["hamming"]

    return {
        "lpips" : str(lpips_score),
        "phash_hamming" : str(phash_delta),
        "pdq_hamming" : str(pdq_delta)
    }