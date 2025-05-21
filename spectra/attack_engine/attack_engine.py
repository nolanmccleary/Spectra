import lpips
from spectra.deltagrad import NES_Signed_Optimizer, NES_Optimizer
from spectra.hashes import Hash_Wrapper
from PIL import Image
from spectra.utils import get_rgb_tensor, rgb_to_grayscale, rgb_to_luma, tensor_resize, inverse_delta, lpips_rgb, to_hex, bool_tensor_delta, byte_quantize, l2_delta, make_acceptance_func
import torch
from torchvision.transforms import ToPILImage

#TODO: Handle fail mode tracking better

DEFAULT_SCALE_FACTOR = 6
#DEFAULT_NUM_PERTURBATIONS = 3000
ALPHA = 2.9
BETA = 0.9 #Hah, Beta.
STEP_COEFF = 0.008


torch.set_default_dtype(torch.float64)

class Attack_Engine:

    def __init__(self, verbose):
        self.attacks = []
        self.image_batch = []
        self.verbose = verbose


    def log(self, msg):
        if self.verbose == "on":
            print(msg)


    def add_attack(self, image_batch: list[tuple[str]], hash_wrapper: Hash_Wrapper, hamming_threshold: int, acceptance_func, attack_cycles: int, device: str, **kwargs):
        self.attacks.append(Attack_Object(hash_wrapper, hamming_threshold, acceptance_func, attack_cycles, device, **kwargs))
        self.image_batch = image_batch


    def run_attacks(self):
        for attack in self.attacks:
            for image in self.image_batch:
                self.log(attack.run_attack(image[0], image[1]))



class Attack_Object:

    def __init__(self, hash_wrapper: Hash_Wrapper, hamming_threshold, acceptance_func, attack_cycles, device, verbose="off"):
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
        self.acceptance_func = make_acceptance_func(self, acceptance_func)
        self.attack_cycles = attack_cycles
        self.resize_flag = True if self.resize_height > 0 and self.resize_width > 0 else False  #Provide resize parameters if your hash pipeline requires resizing

        self.lpips_func = lpips.LPIPS(net='alex').to(self.device)

        self.func_package = (self.func, bool_tensor_delta, byte_quantize)
        self.device_package = (self.func_device, self.device, self.device)


        self.is_staged = False
        


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
        self.rgb_tensor = None
        self.tensor = None
        self.original_hash = None       #Yes I know this is Satanic
        self.current_hash = None
        self.current_hamming = None
        self.optimizer = None
        self.is_staged = False
        self.original_height = None
        self.original_width = None

        self.output_tensor = None 
        self.output_hash = self.current_hash
        self.output_hamming = self.current_hamming
        self.output_lpips = 1
        self.output_l2 = 1
        self.attack_success = None

        self.prev_step = None
        
        
        self.log("Staging attack...\n")
        self.set_tensor(input_image_path)
        
        #self.optimizer = NES_Signed_Optimizer(func_package=self.func_package, device_package=self.device_package, tensor=self.tensor, vecMin=0.0, vecMax=1.0)
        self.optimizer = NES_Optimizer(func_package=self.func_package, device_package=self.device_package, tensor=self.tensor, vecMin=0.0, vecMax=1.0)

        self.num_pertubations = ALPHA
        for k in self.tensor.shape:
            self.num_pertubations *= k
        self.num_pertubations = (int(self.num_pertubations) // 2) * 2

        print(self.num_pertubations)

        self.is_staged = True
        


    #For now we'll just use simple attack logic
    def run_attack(self, input_image_path, output_image_path):
        self.attack_success = False
        if self.is_staged == False:
            self.stage_attack(input_image_path)
        
        self.log("Running attack...\n")

        optimal_delta = None



        optimal_delta = self.optimizer.get_delta(
            step_coeff=STEP_COEFF, 
            num_steps=self.attack_cycles, 
            perturbation_scale_factor=DEFAULT_SCALE_FACTOR,
            num_perturbations=self.num_pertubations, 
            beta=BETA, acceptance_func=self.acceptance_func)



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

            EPS = 1e-6
            for scale in scale_factors:
                cand_delta  = rgb_delta * scale
                cand_tensor = (self.rgb_tensor + cand_delta)

                pos_scale = torch.where(
                    cand_delta > 0,
                (1.0 - self.rgb_tensor) / (cand_delta + EPS),
                torch.tensor(1.0, device=self.device),
                )
                neg_scale = torch.where(
                    cand_delta < 0,
                    (0.0 - self.rgb_tensor) / (cand_delta - EPS),
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



            self.output_lpips = lpips_rgb(self.rgb_tensor, self.output_tensor, self.lpips_func)
            self.output_l2 = l2_delta(self.rgb_tensor, self.output_tensor)
            self.attack_success = True
            out = self.output_tensor.detach()#.cpu()
            output_image = ToPILImage()(out)
            output_image.save(output_image_path)
            self.log(f"Saved attacked image to {output_image_path}")
        
        
        self.is_staged = False

        self.log(f"Success status: {self.attack_success}")
        


        return {
            "success": self.attack_success,
            "original_hash" : to_hex(self.original_hash),
            "output_hash": to_hex(self.output_hash) if self.output_hash is not None else None,
            "hamming_distance": self.output_hamming,
            "lpips": self.output_lpips,
            "l2" : self.output_l2,
        }