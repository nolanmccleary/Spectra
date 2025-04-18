from gradient_engine import Grayscale_Engine, RGB_Engine
from PIL import Image
from utils import get_rgb_tensor
import torch.nn.functional as F



class Attack_Engine:

    #TODO: Figure out how to handle non-square images
    def __init__(self, func, resize_height = -1, resize_width = -1, mode="grayscale", device="cpu", verbose="off"):
        self.valid_modes = {"grayscale", "rgb"}
        valid_devices = {"cpu", "cuda"}
        valid_verbosities = {"on", "off"}

        if mode not in self.valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of: {self.valid_modes}")

        if device not in self.valid_devices:
            raise ValueError(f"Invalid device '{device}'. Expected one of: {valid_devices}")

        if verbose not in valid_verbosities:
            raise ValueError(f"Invalid verbosity '{verbose}'. Expected one of: {valid_verbosities}")

        self.func = func
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.mode = mode
        self.device = device
        self.verbose = verbose
        self.resize_flag = True if (resize_height > 0 and resize_width > 0) else False
        self.image_path = None #Set null pointers for assignment later
        self.tensor = None 



    def log(self, msg):
        if self.verbose == "on":
            print(msg)


    
    def set_tensor(self):
        if self.mode in self.valid_modes:
            with Image.open(self.image_path) as img:
                rgb_tensor = get_rgb_tensor(img, self.device) #[C, H, W]
                
                if self.mode == "grayscale":
                    self.log("Setting grayscale image tensor")

                    gray = rgb_tensor.mean(dim=0, keepdim=True)  #[1, H, W]
                    gray = gray.unsqueeze(0)  #[1, 1, H, W]

                    if self.resize_flag:
                        gray = F.interpolate(   #[1, 1, H_r, W_r]
                            gray,
                            size=(self.resize_height, self.resize_width),
                            mode='bilinear',
                            align_corners=False
                        )

                    self.tensor = gray.view(-1).to(self.device)             #[H x W] or [H_r x W_r]


                elif self.mode == "rgb":
                    self.log("Setting RGB image tensor")

                    if self.resize_flag:
                        rgb_tensor = rgb_tensor.unsqueeze(0) #[1, C, H, W]
    
                        rgb_resized = F.interpolate(
                            rgb_tensor,
                            size=(self.resize_height, self.resize_width),
                            mode='bilinear',
                            align_corners=False
                        )
                        rgb_tensor = rgb_resized.squeeze(0) #[1, C, H_r, W_r] -> [C, H_r, W_r]
                    
                    self.tensor = rgb_tensor.to(self.device) #[1, C, H_r, W_r] or [C, H, W]
                    
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Expected one of: {self.valid_modes}")
        
