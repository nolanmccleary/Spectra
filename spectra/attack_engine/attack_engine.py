import json
import lpips
from spectra.deltagrad import NES_Signed_Optimizer, NES_Optimizer
from spectra.deltagrad.utils import anal_clamp
from spectra.hashes import Hash_Wrapper
from spectra.validation import image_compare
from PIL import Image
from spectra.utils import get_rgb_tensor, rgb_to_grayscale, rgb_to_luma, tensor_resize, inverse_delta, to_hex, bool_tensor_delta, byte_quantize, l2_delta, make_acceptance_func
import torch
from torchvision.transforms import ToPILImage


# TODO: Handle fail mode tracking better
class Attack_Engine:

    def __init__(self, verbose):
        self.attacks = {}
        self.attack_log = {}
        self.verbose = verbose


    def log(self, msg):
        if self.verbose == "on":
            print(msg)


    def add_attack(self, attack_tag, images: list[str], input_image_dirname, output_image_dirname, hash_wrapper: Hash_Wrapper, hyperparameter_set: dict, hamming_threshold: int, acceptance_func, num_reps: int, attack_cycles: int, device: str, lpips_func, **kwargs):
        self.attacks[attack_tag] = (images, input_image_dirname, output_image_dirname, Attack_Object(hash_wrapper, hyperparameter_set, hamming_threshold, acceptance_func, num_reps, attack_cycles, device, lpips_func, self.verbose, **kwargs))


    def run_attacks(self, output_name="spectra_out"):
        # run each registered attack
        for attack_tag in self.attacks.keys():
            self.attack_log[attack_tag] = {}
            for image in self.attacks[attack_tag][0]:
                input_image = f"{self.attacks[attack_tag][1]}/{image}"
                output_image = f"{self.attacks[attack_tag][2]}/{attack_tag}_{image}"
                self.attack_log[attack_tag][image] = self.attacks[attack_tag][3].run_attack(input_image, output_image)

        json_filename = f"{output_name}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.attack_log, f, indent=4)
        print(f"Attack log saved to {json_filename}")




class Attack_Object:

    def __init__(self, hash_wrapper: Hash_Wrapper, hyperparameter_set: dict, hamming_threshold, acceptance_func, num_reps, attack_cycles, device, lpips_func=None, verbose="off", delta_scaledown=False):
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
        self.quant_func = byte_quantize
        
        self.num_reps = num_reps
        self.attack_cycles = attack_cycles

        self.resize_flag = True if self.resize_height > 0 and self.resize_width > 0 else False

        if lpips_func != None:
            self.lpips_func = lpips_func
        else:
            self.lpips_func = lpips.LPIPS(net='alex').to("cpu")

        self.alpha, self.beta, self.step_coeff, self.scale_factor = hyperparameter_set["alpha"], hyperparameter_set["beta"], hyperparameter_set["step_coeff"], hyperparameter_set["scale_factor"]

        self.func_package = (self.func, bool_tensor_delta, self.quant_func)
        self.device_package = (self.func_device, self.device, self.device)

        self.delta_scaledown = delta_scaledown


    def log(self, msg):
        if self.verbose == "on":
            print(msg)



    def set_tensor(self, input_image_path):
        with Image.open(input_image_path) as img:
            self.rgb_tensor = get_rgb_tensor(img, self.device)
            self.original_height = self.rgb_tensor.size(1)
            self.original_width = self.rgb_tensor.size(2)
            self.log("Setting grayscale image tensor")

            self._tensor = self.rgb_tensor

            if self.colormode == "grayscale":
                self._tensor = rgb_to_grayscale(self.rgb_tensor)
            elif self.colormode == "luma":
                self._tensor = rgb_to_luma(self.rgb_tensor)

            if self.resize_flag:
                self._tensor = tensor_resize(self._tensor, self.resize_height, self.resize_width)
                self.height = self.resize_height
                self.width = self.resize_width
            else:
                self.height = self.original_height
                self.width = self.original_width

            self.original_hash = self.func(self._tensor.to(self.func_device))



    def stage_attack(self, input_image_path):
        # reset state
        self.rgb_tensor = None
        self._tensor = None
        self.original_hash = None
        self.optimizer = None
        self.is_staged = False
        self.original_height = None
        self.original_width = None

        self.output_tensor = None
        
        self.output_hash = None
        self.output_hamming = None
        self.output_lpips = 1
        self.output_l2 = 1
        
        self.current_hash = None
        self.current_hamming = None
        self.current_lpips = 1
        self.current_l2 = 1

        self.attack_success = False
        self.prev_step = None

        self.system_state = []

        self.log("Staging attack...\n")
        self.set_tensor(input_image_path)

        self.optimizer = NES_Optimizer(func_package=self.func_package, device_package=self.device_package, tensor=self._tensor, vecMin=0.0, vecMax=1.0)

        # calculate number of perturbations
        self.num_pertubations = self.alpha
        for k in self._tensor.shape:
            self.num_pertubations *= k
        self.num_pertubations = (int(self.num_pertubations) // 2) * 2



    def run_attack(self, input_image_path, output_image_path):
        self.stage_attack(input_image_path)
        self.log("Running attack...\n")

        optimal_delta = None

        for _ in range(self.num_reps):
            curr_delta, accepted = self.optimizer.get_delta(
            step_coeff=self.step_coeff,
            num_steps=self.attack_cycles,
            perturbation_scale_factor=self.scale_factor,
            num_perturbations=self.num_pertubations,
            beta=self.beta, acceptance_func=self.acceptance_func)
            
            if optimal_delta is None or accepted:
                optimal_delta = curr_delta

            
        ################################ RTQ - FROM HASH SPACE TO IMAGE SPACE #####################
        upsampled_delta = optimal_delta
        
        if self.resize_flag:
            optimal_delta = optimal_delta.view(3 if self.colormode == "rgb" else 1, self.height, self.width)
            upsampled_delta = tensor_resize(optimal_delta, self.original_height, self.original_width)
        rgb_delta = upsampled_delta
        if self.colormode in {"grayscale", "luma"}:
            rgb_delta = inverse_delta(self.rgb_tensor, upsampled_delta)

        self.output_tensor = self.rgb_tensor + rgb_delta
        cand_targ = self.output_tensor


        ################################ RTQ - IMAGE SPACE BACK TO HASH SPACE  ########################################

        if self.colormode == "grayscale":           #Adjust target if our hash function requires grayscale or resize
            cand_targ = rgb_to_grayscale(cand_targ)
        elif self.colormode == "luma":
            cand_targ = rgb_to_luma(cand_targ)
        
        if self.resize_flag:
            cand_targ = tensor_resize(cand_targ, self.resize_height, self.resize_width)
    
        
        ################################ END OF ROUND-TRIP QUANTIZATION #####################################

        self.output_hash = self.func(self.quant_func(cand_targ))
        self.output_hamming = self.original_hash.ne(self.output_hash).sum().item()
        self.attack_success = self.output_hamming >= self.hamming_threshold

        ################################# DELTA SCALEDOWN (OPTIONAL) #############################################

        if self.delta_scaledown:
            scale_factors = torch.linspace(0.0, 1.0, steps=50)
            
            for scale in scale_factors:
                cand_delta = rgb_delta * scale
                safe_scale = anal_clamp(self.rgb_tensor, cand_delta, 0.0, 1.0)
                safe_delta = cand_delta * safe_scale
                
                cand_tensor = self.rgb_tensor + safe_delta
                cand_targ = cand_tensor.clone()

                if self.colormode == "grayscale":           #Adjust target if our hash function requires grayscale or resize
                    cand_targ = rgb_to_grayscale(cand_targ)
                elif self.colormode == "luma":
                    cand_targ = rgb_to_luma(cand_targ)
                if self.resize_flag:
                    cand_targ = tensor_resize(cand_targ, self.resize_height, self.resize_width)

                cand_tensor = self.quant_func(cand_tensor)
                cand_targ = self.quant_func(cand_targ)

                cand_hash = self.func(cand_targ.to(self.func_device))
                cand_ham = cand_hash.ne(self.original_hash).sum().item()
                if cand_ham >= self.hamming_threshold:
                    self.output_tensor = cand_tensor
                    self.output_hash = cand_hash
                    self.output_hamming = cand_ham
                    self.attack_success = True
                    break


        ################################# END OF DELTA SCALEDOWN  ###################################################

        self.output_lpips = self.lpips_func(self.rgb_tensor, self.output_tensor)
        self.output_l2 = l2_delta(self.rgb_tensor, self.output_tensor)
        
        out = self.output_tensor.detach()
        output_image = ToPILImage()(out)
        output_image.save(output_image_path)
        
        self.log(f"Saved attacked image to {output_image_path}")

        self.log(f"Success status: {self.attack_success}")
        #self.log(self.system_state)

        return {
            "pre_validation": {
                "success"           : self.attack_success,
                "original_hash"     : to_hex(self.original_hash),
                "output_hash"       : to_hex(self.output_hash) if self.output_hash is not None else None,
                "hamming_distance"  : self.output_hamming,
                "lpips"             : self.output_lpips,
                "l2"                : self.output_l2
            },
            "post_validation": image_compare(input_image_path, output_image_path, self.lpips_func, self.device, self.verbose)
        }