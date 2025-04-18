#TODO: RGB Engine for generic function passing
#TODO: LPIPS instead of L2
#TODO: Need to implement some sort of perturbation scaling algo


import torch
import torch.nn.functional as Func
from PIL import Image
from torchvision.transforms import ToPILImage
import utils 

device = torch.device("cpu")


# Main attack function
def phash_attack():
    NUM_PERTURBATIONS = 3000
    NUM_CYCLES = 40
    NUM_PERTURBATION_GENERATIONS = 5
    INPUT_IMAGE_PATH = 'sample_images/peppers.png'
    OUTPUT_IMAGE_PATH = 'output/peppers_attacked.png'
    DCT_DIM = 8
    DCT_HFF = 4
    DCT_SIDE_LENGTH = DCT_DIM * DCT_HFF
    SCALE_FACTOR = 6.0
    STEP_SIZE = 0.01
    HASH_THRESHOLD = 5
    SEED_CONST = 0.06


    rgb_image = Image.open(INPUT_IMAGE_PATH)
    rgb_tensor = utils.get_rgb_tensor(rgb_image, device)
    grayscale_tensor = utils.get_grayscale_tensor(rgb_tensor, DCT_SIDE_LENGTH, device)

    initial_hash = utils.utils.generate_phash(grayscale_tensor, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)

    dim = grayscale_tensor.numel()
    optimal_delta = torch.zeros_like(grayscale_tensor)
    min_l2 = 1



    for _ in range(NUM_PERTURBATION_GENERATIONS):
        perturbations = utils.generate_perturbation_vectors_1d(dim, NUM_PERTURBATIONS // 2, device)
        current_image = grayscale_tensor.clone()
        total_delta = torch.zeros_like(current_image)
        seed_flag = True


        for _ in range(NUM_CYCLES):
            if seed_flag:
                current_image = grayscale_tensor.clone()
                current_image.add_(utils.generate_seed_perturbation(dim, SEED_CONST, device).squeeze(0)).clamp(0.0, 1.0)
                seed_flag = False
            
            
            gradient = torch.zeros_like(current_image)
            ph_curr = utils.utils.generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
            

            for i in range(NUM_PERTURBATIONS):
                scaled_pert = perturbations[i] * SCALE_FACTOR
                h2 = utils.generate_phash(current_image + scaled_pert, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)
                delta_ham = utils.hamming_distance_hex(ph_curr, h2)
                gradient.add_(delta_ham * scaled_pert)
            
            delta = gradient.sign() * STEP_SIZE
            current_image = (current_image + delta).clamp(0.0, 1.0)
            total_delta.add_(delta)
            ham = utils.hamming_distance_hex(utils.generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM), initial_hash)
            
            l2 = utils.l2_per_pixel_grayscale_1d(grayscale_tensor, current_image)
            print(l2, min_l2)

            if ham >= HASH_THRESHOLD:
                if l2 < min_l2:
                    min_l2 = l2
                    optimal_delta = total_delta.clone()
                
                total_delta = torch.zeros_like(current_image)
                seed_flag = True 
                


    grayscale_hash = utils.generate_phash(current_image, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH, DCT_DIM)


    small = optimal_delta.view(1, 1, DCT_SIDE_LENGTH, DCT_SIDE_LENGTH)
    C, H, W = rgb_tensor.shape
    upsampled_delta = Func.interpolate(
        small,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).view(-1)
    image_delta = utils.inverse_delta(rgb_tensor, upsampled_delta)

    output_tensor, output_hash, output_hamming, output_l2 = utils.minimize_rgb_l2_preserve_hash(rgb_tensor, image_delta, initial_hash, grayscale_hash, DCT_SIDE_LENGTH, DCT_DIM, HASH_THRESHOLD, device)

    # Print results
    print("Initial Hash:", initial_hash)
    print("Final Hash:  ", output_hash)
    print("Final  Hamming Distance:   ", output_hamming)
    print("Final RGB L2:", output_l2)

    # Save image
    out = output_tensor.detach().cpu()
    output_image = ToPILImage()(out)
    output_image.save(OUTPUT_IMAGE_PATH)
    print(f"Saved attacked image to {OUTPUT_IMAGE_PATH}")







if __name__ == '__main__':
    phash_attack()