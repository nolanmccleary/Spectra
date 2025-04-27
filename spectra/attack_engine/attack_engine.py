import lpips
from spectra.gradient_engine import make_gradient_engine
from spectra.hashes import Hash_Wrapper
from PIL import Image
from spectra.utils import get_rgb_tensor, rgb_to_grayscale, hamming_distance_hex, grayscale_resize_and_flatten, inverse_delta, l2_per_pixel_rgb
import torch
from torchvision.transforms import ToPILImage



DEFAULT_SCALE_FACTOR = 6
DEFAULT_NUM_PERTURBATIONS = 3000

class Attack_Engine:

    def __init__(self, verbose):
        self.attacks = []
        self.image_batch = []
        self.verbose = verbose


    def log(self, msg):
        if self.verbose == "on":
            print(msg)


    def add_attack(self, image_batch: list[tuple[str]], hash_wrapper: Hash_Wrapper, hamming_threshold: int, attack_cycles: int, device: str, **kwargs):
        self.attacks.append(Attack_Object(hash_wrapper, hamming_threshold, attack_cycles, device, **kwargs))
        self.image_batch = image_batch


    def run_attacks(self):
        for attack in self.attacks:
            for image in self.image_batch:
                self.log(attack.run_attack(image[0], image[1]))



class Attack_Object:

    def __init__(self, hash_wrapper: Hash_Wrapper, hamming_threshold, attack_cycles, device, verbose="off"):
        valid_devices = {"cpu", "cuda", "mps"}
        valid_verbosities = {"on", "off"}
        if device not in valid_devices:
            raise ValueError(f"Invalid device '{device}'. Expected one of: {valid_devices}")
        if verbose not in valid_verbosities:
            raise ValueError(f"Invalid verbosity '{verbose}'. Expected one of: {valid_verbosities}")
                
        self.device = device
        self.verbose = verbose

        self.func, self.resize_height, self.resize_width, available_devices = hash_wrapper.get_info()     
        if device in available_devices:
            self.func_device = device
        else:
            self.log(f"Warning, current hash function '{hash_wrapper.get_name()}' does not support the chosen device {device}. Defaulting to CPU for hash function calls; this will add overhead.")
            self.func_device = "cpu"

        self.hamming_threshold = hamming_threshold
        self.attack_cycles = attack_cycles
        self.resize_flag = True if self.resize_height > 0 and self.resize_width > 0 else False  #Provide resize parameters if your hash pipeline requires resizing

        self.rgb_tensor = None
        self.tensor = None
        self.original_hash = None 
        self.current_hash = None
        self.current_hamming = None
        self.gradient_engine = None 
        self.is_staged = False
        self.original_height = None
        self.original_width = None

        self.output_tensor = None 
        self.output_hash = None 
        self.output_hamming = 0
        self.output_l2 = 1
        self.attack_success = None

        self.loss_func = lpips.LPIPS(net='alex').to(self.device)



    def log(self, msg):
        if self.verbose == "on":
            print(msg)



    def set_tensor(self, input_image_path):
        with Image.open(input_image_path) as img:
            self.rgb_tensor = get_rgb_tensor(img, self.device) #[C, H, W]
            self.original_height = self.rgb_tensor.size(1)
            self.original_width = self.rgb_tensor.size(2)
            self.log("Setting grayscale image tensor")
            gray = rgb_to_grayscale(self.rgb_tensor)

            if self.resize_flag:
                gray = grayscale_resize_and_flatten(gray, self.resize_height, self.resize_width) 
                self.height = self.resize_height
                self.width = self.resize_width

            else:
                self.height = self.original_height
                self.width = self.original_width
            
            self.tensor = gray
            self.original_hash = self.func(self.tensor.to(self.func_device), self.height, self.width)  #If the hash func resizes/grayscales, we allow the option of an upfront conversion to save compute on every function call during the attack
            self.current_hash = self.original_hash


    def stage_attack(self, input_image_path):
        self.log("Staging attack...\n")
        self.set_tensor(input_image_path)
        self.gradient_engine = make_gradient_engine(self.func, self.tensor, self.device, self.func_device, DEFAULT_NUM_PERTURBATIONS, self.height, self.width, self.loss_func)
        self.is_staged = True
        


    #For now we'll just use simple attack logic
    def run_attack(self, input_image_path, output_image_path, step_size = 0.01):
        self.attack_success = False
        if self.is_staged == False:
            self.stage_attack(input_image_path)
        
        self.log("Running attack...\n")

        current_delta = torch.zeros_like(self.tensor)
        optimal_delta = None
        
        
        #Attack loop
        for _ in range(self.attack_cycles):
            last_tensor_hash = torch.tensor(self.current_hash, dtype=torch.uint64, device=self.device) # h_old -> [h_old]
            step = torch.sign(self.gradient_engine.compute_gradient(last_tensor_hash, DEFAULT_SCALE_FACTOR)) * step_size #Might be better to just get signed gradient 
            
            
            current_delta.add_(step)
            self.gradient_engine.tensor.add_(step).clamp_(0.0, 1.0)
            self.current_hash = self.func(self.gradient_engine.tensor.to(self.func_device), self.height, self.width)
            
            self.current_hamming = hamming_distance_hex(self.original_hash, self.current_hash)


            if self.current_hamming >= self.hamming_threshold:
                l2_distance = self.gradient_engine.l2_delta_from_engine_tensor(self.tensor)

                if l2_distance < self.output_l2:
                    optimal_delta = current_delta.clone()
                    self.output_l2 = l2_distance
                    self.output_hamming = self.current_hamming
                    self.output_hash = self.current_hash
                
                else:   #L2 distance more or less increases monotonically so once we know it isn't better than our current best we may as well re-start; <- NEED TO TEST THIS
                    current_delta.zero_()
                    self.gradient_engine.tensor = self.tensor.clone()
                    


        #If we broke hamming threshold
        if optimal_delta is not None:
            
            upsampled_delta = optimal_delta
            
            if self.resize_flag:               
                optimal_delta = optimal_delta.view(1, self.height, self.width)
                upsampled_delta = grayscale_resize_and_flatten(optimal_delta, self.original_height, self.original_width)

            rgb_delta = inverse_delta(self.rgb_tensor, upsampled_delta)

            #Backwards pass to optimize RGB L2 delta within hamming threshold constraints
            scale_factors = torch.linspace(0.0, 1.0, steps=50)
            self.output_tensor = (self.rgb_tensor + rgb_delta)

            for scale in scale_factors:
                cand_delta  = rgb_delta * scale
                cand_tensor = (self.rgb_tensor + cand_delta)

                eps = 1e-6
                pos_scale = torch.where(
                    cand_delta > 0,
                   (1.0 - self.rgb_tensor) / (cand_delta + eps),
                   torch.tensor(1.0, device=self.device),
                )
                neg_scale = torch.where(
                    cand_delta < 0,
                    (0.0 - self.rgb_tensor) / (cand_delta - eps),
                    torch.tensor(1.0, device=self.device),
                )
                safe_scale = torch.min(pos_scale, neg_scale).clamp(max=1.0)
                safe_delta = cand_delta * safe_scale
                cand_tensor = self.rgb_tensor + safe_delta


                cand_gray = rgb_to_grayscale(cand_tensor)
                
                if self.resize_flag:
                    cand_gray = grayscale_resize_and_flatten(cand_gray, self.resize_height, self.resize_width)


                cand_hash = self.func(cand_gray.to(self.func_device), self.height, self.width)
                cand_ham = hamming_distance_hex(cand_hash, self.original_hash)

                if cand_ham >= self.hamming_threshold:
                    self.output_tensor = cand_tensor
                    self.output_hamming = cand_ham
                    self.output_hash = cand_hash
                    break
                else:
                    continue

            self.output_l2 = l2_per_pixel_rgb(self.rgb_tensor, self.output_tensor, self.loss_func)
            self.attack_success = True
            out = self.output_tensor.detach().cpu()
            output_image = ToPILImage()(out)
            output_image.save(output_image_path)
            self.log(f"Saved attacked image to {output_image_path}")
        
        
        self.is_staged = False
        

        self.log(f"Success status: {self.attack_success}")
        
        if self.attack_success:
            self.log(f"Original hash: {hex(self.original_hash)}")
            self.log(f"Current hash: {hex(self.output_hash)}")
            self.log(f"Final hash hamming distance: {self.output_hamming}")
            self.log(f"Final L2 distance: {self.output_l2}")


        return {
            "success": self.attack_success,
            "original_hash" : hex(self.original_hash),
            "output_hash": hex(self.output_hash) if self.output_hash is not None else None,
            "hamming_distance": self.output_hamming,
            "l2_per_pixel": self.output_l2,
        }