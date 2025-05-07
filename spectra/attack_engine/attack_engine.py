import lpips
from spectra.deltagrad import Gradient_Engine
from spectra.hashes import Hash_Wrapper
from PIL import Image
from spectra.utils import get_rgb_tensor, rgb_to_grayscale, rgb_to_luma, tensor_resize, inverse_delta, lpips_rgb, to_hex, bool_tensor_delta, byte_quantize, lpips_delta_from_engine_tensor
import torch
from torchvision.transforms import ToPILImage



DEFAULT_SCALE_FACTOR = 6
DEFAULT_NUM_PERTURBATIONS = 3000
BETA = 0.85 #Hah, Beta.



torch.set_default_dtype(torch.float64)

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

        self.func, self.resize_height, self.resize_width, available_devices, self.colormode = hash_wrapper.get_info()     
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
        self.output_lpips = 1
        self.attack_success = None

        self.prev_step = None

        self.loss_func = lpips.LPIPS(net='alex').to(self.device)



    def log(self, msg):
        if self.verbose == "on":
            print(msg)



    def set_tensor(self, input_image_path):
        with Image.open(input_image_path) as img:
            self.rgb_tensor = get_rgb_tensor(img, self.device) #[C, H, W]; R - [0.0, 1.0]
            self.original_height = self.rgb_tensor.size(1)
            self.original_width = self.rgb_tensor.size(2)
            self.log("Setting grayscale image tensor")
            
            self.tensor = self.rgb_tensor

            if self.colormode == "grayscale":
                self.tensor = rgb_to_grayscale(self.rgb_tensor)

            elif self.colormode == "luma":
                self.tensor = rgb_to_luma(self.rgb_tensor)

            if self.resize_flag:
                self.tensor = tensor_resize(self.tensor, self.resize_height, self.resize_width) 
                self.height = self.resize_height
                self.width = self.resize_width
            else:
                self.height = self.original_height
                self.width = self.original_width

            self.original_hash = self.func(self.tensor.to(self.func_device))  #If the hash func resizes/grayscales, we allow the option of an upfront conversion to save compute on every function call during the attack
            self.current_hash = self.original_hash



    def stage_attack(self, input_image_path):
        self.log("Staging attack...\n")
        self.set_tensor(input_image_path)
        self.gradient_engine = Gradient_Engine(
            func=self.func, 
            tensor=self.tensor, 
            device=self.device, 
            func_device=self.func_device, 
            num_perturbations=DEFAULT_NUM_PERTURBATIONS, 
            delta_func=bool_tensor_delta,
            quant_func=byte_quantize)
       
       
        self.is_staged = True
        


    #For now we'll just use simple attack logic
    def run_attack(self, input_image_path, output_image_path, step_size = 0.005):
        self.attack_success = False
        if self.is_staged == False:
            self.stage_attack(input_image_path)
        
        self.log("Running attack...\n")

        current_delta = torch.zeros_like(self.tensor)
        optimal_delta = None
        
        eps = 1e-6

        #Attack loop
        for _ in range(self.attack_cycles):
            last_tensor_hash = torch.tensor(self.current_hash, dtype=torch.bool, device=self.device) # h_old -> [h_old]
            step = torch.sign(self.gradient_engine.compute_gradient(DEFAULT_SCALE_FACTOR, 0.0, 1.0)) * step_size * BETA #Might be better to just get signed gradient 
            
            if self.prev_step is not None:
                step.add_((1 - BETA) * self.prev_step)
            

            #Adaptively scale step to avoid disrupting image composition via clipping
            pos_scale = torch.where(
                step > 0,
                (1.0 - self.gradient_engine.tensor) / (step + eps),
                torch.tensor(1.0, device=self.device),
            )
            neg_scale = torch.where(
                step < 0,
                (0.0 - self.gradient_engine.tensor) / (step - eps),
                torch.tensor(1.0, device=self.device),
            )

            safe_scale = torch.min(pos_scale, neg_scale).clamp(max=1.0)

            delta_step = step * safe_scale
            self.prev_step = delta_step


            current_delta.add_(delta_step)
            self.gradient_engine.tensor.add_(delta_step)#.clamp_(0.0, 1.0)


            self.current_hash = self.func(self.gradient_engine.tensor.to(self.func_device))
            self.current_hamming = int((self.original_hash != self.current_hash).sum().item())


            if self.current_hamming >= self.hamming_threshold:
                lpips_distance = lpips_delta_from_engine_tensor(self.tensor, self.gradient_engine.tensor, self.loss_func)

                if lpips_distance < self.output_lpips:
                    optimal_delta = current_delta.clone()
                    self.output_lpips = lpips_distance
                    self.output_hamming = self.current_hamming
                    self.output_hash = self.current_hash
                
                else:   #Lpips distance more or less increases monotonically so once we know it isn't better than our current best we may as well re-start; <- NEED TO TEST THIS
                    current_delta.zero_()
                    self.gradient_engine.tensor = self.tensor.clone()


        #If we broke hamming threshold
        if optimal_delta is not None:
            
            upsampled_delta = optimal_delta
            
            if self.resize_flag:               
                optimal_delta = optimal_delta.view(3 if self.colormode == "rgb" else 1, self.height, self.width)
                upsampled_delta = tensor_resize(optimal_delta, self.original_height, self.original_width)

            rgb_delta = upsampled_delta


            if self.colormode == "grayscale" or self.colormode == "luma":
                rgb_delta = inverse_delta(self.rgb_tensor, upsampled_delta)



            #Backwards pass to optimize RGB Lpips delta within hamming threshold constraints
            scale_factors = torch.linspace(0.0, 1.0, steps=50)
            self.output_tensor = (self.rgb_tensor + rgb_delta)


            for scale in scale_factors:
                cand_delta  = rgb_delta * scale
                cand_tensor = (self.rgb_tensor + cand_delta)

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

                cand_targ = cand_tensor
                if self.colormode == "grayscale":
                    cand_targ = rgb_to_grayscale(cand_tensor)
                
                elif self.colormode == "luma":
                    cand_targ = rgb_to_luma(cand_tensor)

                if self.resize_flag:
                    cand_targ = tensor_resize(cand_targ, self.resize_height, self.resize_width)


                cand_hash = self.func(cand_targ.to(self.func_device))
                cand_ham = cand_hash.ne(self.original_hash).sum().item()

                if cand_ham >= self.hamming_threshold:
                    self.output_tensor = cand_tensor
                    self.output_hamming = cand_ham
                    self.output_hash = cand_hash
                    break
                else:
                    continue



            self.output_lpips = lpips_rgb(self.rgb_tensor, self.output_tensor, self.loss_func)
            self.attack_success = True
            out = self.output_tensor.detach()#.cpu()
            output_image = ToPILImage()(out)
            output_image.save(output_image_path)
            self.log(f"Saved attacked image to {output_image_path}")
        
        
        self.is_staged = False
        

        self.log(f"Success status: {self.attack_success}")
        
        if self.attack_success:
            self.log(f"Original hash: {to_hex(self.original_hash)}")
            self.log(f"Current hash: {to_hex(self.output_hash)}")
            self.log(f"Final hash hamming distance: {self.output_hamming}")
            self.log(f"Final Lpips distance: {self.output_lpips}")


        return {
            "success": self.attack_success,
            "original_hash" : to_hex(self.original_hash),
            "output_hash": to_hex(self.output_hash) if self.output_hash is not None else None,
            "hamming_distance": self.output_hamming,
            "lpips": self.output_lpips,
        }