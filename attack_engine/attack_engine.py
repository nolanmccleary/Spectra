from gradient_engine import make_gradient_engine
from PIL import Image
from utils import get_rgb_tensor, hamming_distance_hex, inverse_delta, minimize_rgb_l2_preserve_hash
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

    def __init__(self, func, image_path, attack_cycles=100, resize_height=-1, resize_width=-1, format="grayscale", device="cpu", verbose="off"):
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
        self.attack_cycles = attack_cycles
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.format = format
        self.device = device
        self.verbose = verbose

        self.rgb_tensor = None
        self.tensor = None
        self.original_hash = None 
        self.current_hash = None
        self.gradient_engine = None 
        self.is_staged = False
        self.resize_flag = True if (self.resize_height > 0 and self.resize_width > 0) else False
        self.original_height = None
        self.original_width = None

        self.output_tensor = None 
        self.output_hash = None 
        self.output_hamming = None 
        self.output_l2 = None




    def log(self, msg):
        if self.verbose == "on":
            print(msg)



    def set_tensor(self):
        with Image.open(self.image_path) as img:
            self.rgb_tensor = get_rgb_tensor(img, self.device) #[C, H, W]
            
            self.log("Setting grayscale image tensor")

            gray = self.rgb_tensor.mean(dim=0, keepdim=True)  #[1, H, W]
            gray = gray.unsqueeze(0)  #[1, 1, H, W]

            if self.resize_flag:
                self.original_height = self.rgb_tensor.size(1)
                self.original_width = self.rgb_tensor.size(2)

                gray = F.interpolate(   #[1, 1, H_r, W_r]   TODO: Package this into a function
                    gray,
                    size=(self.resize_height, self.resize_width),
                    mode='bilinear',
                    align_corners=False
                )

            self.tensor = gray.view(-1).to(self.device)      #[H x W] or [H_r x W_r]

            self.original_hash = self.func(self.tensor, device=self.device)  #If the hash func resizes/grayscales, we allow the option of an upfront conversion to save compute on every function call during the attack
            self.current_hash = self.original_hash
    


    def stage_attack(self):
        self.log("Staging attack...\n")
        self.tensor = self.set_tensor()
        self.gradient_engine = make_gradient_engine(self.func, self.tensor, self.device)
        self.attack_set = True
        


    #For now we'll just use simple attack logic
    def run_attack(self, hamming_threshold):
        if self.is_staged == False:
            self.stage_attack()

        self.log("Running attack...\n")

        STEP_SIZE = 0.01
        min_l2 = 1

        optimal_delta = None
        total_delta = torch.zeros_like(self.tensor)
        
        for _ in range(self.attack_cycles):
            step = self.gradient_engine.compute_gradient(self.current_hash) * STEP_SIZE
            total_delta.add_(step)
            self.gradient_engine.tensor.add_(step)
            self.current_hash = self.func(self.gradient_engine.tensor, device=self.device)
            ham = hamming_distance_hex(self.original_hash, self.current_hash)

            if ham >= hamming_threshold:
                l2_distance = self.gradient_engine.l2_per_pixel(self.tensor)

                if l2_distance < min_l2:
                    min_l2 = l2_distance
                    optimal_delta = total_delta
                    total_delta = torch.zeros_like(optimal_delta.clone())
                
                else:   #L2 distance more or less increases monotonically so once we know it isn't better than our current best we may as well re-start
                    self.gradient_engine.tensor = self.tensor.clone()



        if optimal_delta != None:
            if self.resize_flag:
                small_delta = optimal_delta.unsqueeze(0)

                upsampled_delta = F.interpolate(
                    small_delta,
                    size=(self.original_height, self.original_width),
                    mode='bilinear',
                    align_corners=False
                ).view(-1)

            rgb_delta = inverse_delta(self.rgb_tensor, upsampled_delta)

            self.output_tensor, self.output_hash, self.output_hamming, self.output_l2 = minimize_rgb_l2_preserve_hash(self.rgb_tensor, rgb_delta, )

            
                



        self.is_staged = False