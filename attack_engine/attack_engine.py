#TODO: Build func wrapper

from gradient_engine import make_gradient_engine
from PIL import Image
from utils import get_rgb_tensor, rgb_to_grayscale, hamming_distance_hex, grayscale_resize, inverse_delta, l2_per_pixel_rgb
import torch
import torch.nn.functional as F



class Attack_Engine:

    #TODO: Figure out how to handle non-square images
    def __init__(self, verbose="off"):
        self.valid_formats = {"grayscale", "rgb"}
        self.valid_devices = {"cpu", "cuda"}
        valid_verbosities = {"on", "off"}

        if verbose not in valid_verbosities:
            raise ValueError(f"Invalid verbosity '{verbose}'. Expected one of: {valid_verbosities}")

        self.verbose = verbose
        self.attacks = []


    def log(self, msg):
        if self.verbose == "on":
            print(msg)


    def add_attack(self, func, image_path, **kwargs):
        self.attacks.append(Attack_Object(func, image_path, **kwargs))

    
    def run_attacks(self):
        for attack in self.attacks:
            attack.run_attack()



class Attack_Object:

    def __init__(self, func, image_path, hamming_threshold, attack_cycles=100, resize_height=-1, resize_width=-1, format="grayscale", device="cpu", verbose="off"):
        valid_formats = {"grayscale", "rgb"}
        valid_devices = {"cpu", "cuda", "mps"}
        valid_verbosities = {"on", "off"}
        if format not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Expected one of: {valid_formats}")
        if device not in valid_devices:
            raise ValueError(f"Invalid device '{device}'. Expected one of: {valid_devices}")
        if verbose not in valid_verbosities:
            raise ValueError(f"Invalid verbosity '{verbose}'. Expected one of: {valid_verbosities}")
        
        self.func = func                #The function provided here must operate on a grayscaled and resized tensor as well as take a device argument
        self.image_path = image_path
        self.hamming_threshold = hamming_threshold
        self.attack_cycles = attack_cycles
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.format = format
        self.device = device
        self.verbose = verbose
        self.resize_flag = True if self.resize_height < 0 and self.resize_width < 0 else False  #Provide resize parameters if your hash pipeline requires resizing

        self.rgb_tensor = None
        self.tensor = None
        self.original_hash = None 
        self.current_hash = None
        self.current_hamming = None
        self.gradient_engine = None 
        self.is_staged = False
        self.original_height = self.rgb_tensor.size(1)
        self.original_width = self.rgb_tensor.size(2)

        self.output_tensor = None 
        self.output_hash = None 
        self.output_hamming = None 
        self.output_l2 = 1




    def log(self, msg):
        if self.verbose == "on":
            print(msg)



    def set_tensor(self):
        with Image.open(self.image_path) as img:
            self.rgb_tensor = get_rgb_tensor(img, self.device) #[C, H, W]
            self.log("Setting grayscale image tensor")
            gray = rgb_to_grayscale(self.rgb_tensor)

            if self.resize_flag:
                gray = grayscale_resize(gray, self.resize_height, self.resize_width) 

            self.tensor = gray
                
            self.original_hash = self.func(self.tensor, device=self.device)  #If the hash func resizes/grayscales, we allow the option of an upfront conversion to save compute on every function call during the attack
            self.current_hash = self.original_hash
    


    def stage_attack(self):
        self.log("Staging attack...\n")
        self.tensor = self.set_tensor()
        self.gradient_engine = make_gradient_engine(self.func, self.tensor, self.device)
        self.attack_set = True
        


    #For now we'll just use simple attack logic
    def run_attack(self, step_size = 0.01):
        if self.is_staged == False:
            self.stage_attack()

        self.log("Running attack...\n")

        optimal_delta = None
        total_delta = torch.zeros_like(self.tensor)
        
        #Attack loop
        for _ in range(self.attack_cycles):
            step = self.gradient_engine.compute_gradient(self.current_hash) * step_size
            total_delta.add_(step)
            self.gradient_engine.tensor.add_(step)
            self.current_hash = self.func(self.gradient_engine.tensor, device=self.device)
            self.current_hamming = hamming_distance_hex(self.original_hash, self.current_hash)

            if self.current_hamming >= self.hamming_threshold:
                l2_distance = self.gradient_engine.l2_per_pixel(self.tensor)

                if l2_distance < self.output_l2:
                    self.output_l2 = l2_distance
                    optimal_delta = total_delta
                
                else:   #L2 distance more or less increases monotonically so once we know it isn't better than our current best we may as well re-start
                    self.gradient_engine.tensor = self.tensor.clone()
                    total_delta = torch.zeros_like(self.tensor)


        #If we broke hamming threshold
        if optimal_delta != None:
            upsampled_delta = optimal_delta
            
            if self.resize_flag:
                upsampled_delta = grayscale_resize(optimal_delta, self.original_height, self.original_width)

            rgb_delta = inverse_delta(self.rgb_tensor, upsampled_delta)


            #Backwards pass to optimize RGB L2 delta within hamming threshold constraints
            scale_factors = torch.linspace(0.0, 1.0, steps=50)
            
            self.output_tensor = (self.rgb_tensor + rgb_delta)
            self.output_hash = self.current_hash
            self.output_hamming = self.current_hamming

            for scale in scale_factors:
                cand_delta  = rgb_delta * scale
                cand_tensor = (self.rgb_tensor + cand_delta).clamp(0.0, 1.0)

                cand_gray = rgb_to_grayscale(cand_tensor)
                
                if self.resize_flag:
                    cand_gray = grayscale_resize(cand_gray, self.resize_height, self.resize_width)

                cand_hash = self.func(cand_gray, device=self.device)
                cand_ham = hamming_distance_hex(cand_hash, self.original_hash)

                if cand_ham >= self.hamming_threshold:
                    self.output_tensor = cand_tensor
                    self.output_hamming = cand_ham
                    self.output_hash = cand_hash
                    break
                else:
                    continue

            self.output_l2 = l2_per_pixel_rgb(self.rgb_tensor, self.output_tensor)


        self.is_staged = False