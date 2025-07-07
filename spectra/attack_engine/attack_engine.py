import json
import lpips
from spectra.deltagrad import NES_Signed_Optimizer, NES_Optimizer
from spectra.deltagrad.utils import anal_clamp
from spectra.hashes import Hash_Wrapper
from spectra.validation import image_compare
from pathlib import Path
from PIL import Image
from spectra.utils import get_rgb_tensor, tensor_resize, to_hex, bool_tensor_delta, l2_delta, generate_acceptance, generate_conversion, generate_inversion, generate_quant, create_sweep
import torch
from torchvision.transforms import ToPILImage


# TODO: Null guard, fix lack of safe scale wothout delta scaledown.
class Attack_Engine:

    def __init__(self, verbose):
        self.attacks = {}
        self.attack_log = {}
        self.verbose = verbose


    def log(self, msg):
        if self.verbose == "on":
            print(msg)


    def add_attack(self, attack_tag, input_image_dirname, output_image_dirname, *args, **kwargs):
        dir = Path(input_image_dirname)
        images = [f.name for f in dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        self.attacks[attack_tag] = (images, input_image_dirname, output_image_dirname, Attack_Object(*args, **kwargs, verbose=self.verbose))


    def run_attacks(self, output_name="spectra_out"):
        # run each registered attack
        for attack_tag in self.attacks.keys():
            self.attack_log[attack_tag] = {"per_image_results" : {}, "average_results" : {}}
            for image in self.attacks[attack_tag][0]:
                input_image = f"{self.attacks[attack_tag][1]}/{image}"
                output_image = f"{self.attacks[attack_tag][2]}/{attack_tag}_{image}"
                self.attack_log[attack_tag]["per_image_results"][image] = self.attacks[attack_tag][3].run_attack(input_image, output_image)
            
            i = 0
            sum_phash_hamming = 0
            sum_ahash_hamming = 0
            sum_dhash_hamming = 0
            sum_pdq_hamming = 0
            sum_lpips = 0.0
            sum_l2 = 0.0

            sum_beta = 0
            sum_scale_factor = 0
            sum_num_steps = 0

            for image in self.attack_log[attack_tag]["per_image_results"].values():
                if image["pre_validation"]["success"] == True:
                    sum_beta            += float(image["pre_validation"]["ideal_beta"])
                    sum_scale_factor    += float(image["pre_validation"]["ideal_scale_factor"])
                    sum_num_steps       += float(image["pre_validation"]["num_steps"])
                    sum_phash_hamming   += int(image["post_validation"]["phash_hamming"])
                    sum_ahash_hamming   += int(image["post_validation"]["ahash_hamming"])
                    sum_dhash_hamming   += int(image["post_validation"]["dhash_hamming"])
                    sum_pdq_hamming     += int(image["post_validation"]["pdq_hamming"])
                    sum_lpips           += float(image["post_validation"]["lpips"])
                    sum_l2              += float(image["post_validation"]["l2"])
                    i                   += 1


            if i > 0:
                self.attack_log[attack_tag]["average_results"] = {
                    "average_phash_hamming"         : sum_phash_hamming / i,    #TODO: Change naming from 'average' to 'mean'
                    "average_ahash_hamming"         : sum_ahash_hamming / i,
                    "average_dhash_hamming"         : sum_dhash_hamming / i,
                    "average_pdq_hamming"           : sum_pdq_hamming / i,
                    "average_lpips"                 : sum_lpips / i,
                    "average_l2"                    : sum_l2 / i,
                    "average_ideal_beta"            : sum_beta / i,
                    "average_ideal_scale_factor"    : sum_scale_factor / i,
                    "average_num_steps"             : sum_num_steps / i         
                }


        json_filename = f"{output_name}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.attack_log, f, indent=4)
        print(f"Attack log saved to {json_filename}")



#TODO: Refactor input set
class Attack_Object:

    def __init__(self, hash_wrapper: Hash_Wrapper, hyperparameter_set: dict, hamming_threshold, colormode, acceptance_func, quant_func, lpips_func, num_reps, attack_cycles, device, delta_scaledown=False, gate=None, verbose="off"):
        valid_devices = {"cpu", "cuda", "mps"}
        valid_verbosities = {"on", "off"}
        if device not in valid_devices:
            raise ValueError(f"Invalid device '{device}'. Expected one of: {valid_devices}")
        if verbose not in valid_verbosities:
            raise ValueError(f"Invalid verbosity '{verbose}'. Expected one of: {valid_verbosities}")

        self.hamming_threshold = hamming_threshold
        self.colormode = colormode

        self.device = device
        self.verbose = verbose

        self.func, self.resize_height, self.resize_width, available_devices = hash_wrapper.get_info()
        if device in available_devices:
            self.func_device = device
        else:
            self.log(f"Warning, current hash function '{hash_wrapper.get_name()}' does not support the chosen device {device}. Defaulting to CPU for hash function calls; this will add overhead.")
            self.func_device = "cpu"
        
        self.acceptance_func = generate_acceptance(self, acceptance_func)
        
        self.quant_func = generate_quant(quant_func)

        self.num_reps = num_reps
        self.attack_cycles = attack_cycles
        self.resize_flag = True if self.resize_height > 0 and self.resize_width > 0 else False

        if lpips_func != None:
            self.lpips_func = lpips_func
        else:
            self.lpips_func = lpips.LPIPS(net='alex').to("cpu")
        
        self.l2_func = l2_delta

        self.alpha, self.betas, self.step_coeff, self.scale_factors = hyperparameter_set["alpha"], create_sweep(*hyperparameter_set["beta"]), hyperparameter_set["step_coeff"], create_sweep(*hyperparameter_set["scale_factor"])
        self.func_package = (self.func, bool_tensor_delta, self.quant_func)
        self.device_package = (self.func_device, self.device, self.device)
        self.gate = gate
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

            self.inversion_func = generate_inversion(self.colormode)
            self.conversion_func = generate_conversion(self.colormode)

            self._tensor = self.conversion_func(self.rgb_tensor)

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
        self.min_steps = self.attack_cycles

        self.current_hash = None
        self.current_hamming = None
        self.current_lpips = 1
        self.current_l2 = 1

        self.attack_success = False
        self.prev_step = None


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

        min_avg_steps = self.attack_cycles
        ret_set = (None, None, None, None)

        self.log(f"Beta sweep across: {self.betas}\n")
        self.log(f"Perturbation scale factor sweep across: {self.scale_factors}\n")


        for beta in self.betas:
            for scale_factor in self.scale_factors:

                for rep in range(self.num_reps):
                    step_count, curr_delta, accepted = self.optimizer.get_delta(
                    step_coeff=self.step_coeff,
                    num_steps=self.attack_cycles,
                    perturbation_scale_factor=scale_factor,
                    num_perturbations=self.num_pertubations,
                    beta=beta, acceptance_func=self.acceptance_func)
                    
                    if accepted or ret_set[0] is None:          #We get the acceptance best out of our entire sweep space for our output tensor
                        ret_set = (curr_delta, step_count, beta, scale_factor)
                        self.log((ret_set[1], ret_set[2], ret_set[3]))


        ################################ RTQ - FROM HASH SPACE TO IMAGE SPACE #####################
        output_delta = ret_set[0]
        
        if output_delta is not None:

            if self.resize_flag:
                optimal_delta = output_delta.view(3 if self.colormode == "rgb" else 1, self.height, self.width)
                output_delta = tensor_resize(optimal_delta, self.original_height, self.original_width)
            
            rgb_delta = self.inversion_func(self.rgb_tensor, output_delta)
            safe_scale = anal_clamp(self.rgb_tensor, rgb_delta, 0.0, 1.0)

            self.output_tensor = self.rgb_tensor + rgb_delta * safe_scale

            cand_targ = self.conversion_func(self.output_tensor)
            if self.resize_flag:
                cand_targ = tensor_resize(cand_targ, self.resize_height, self.resize_width)

            self.output_hash = self.func(self.quant_func(cand_targ))
            self.output_hamming = self.original_hash.ne(self.output_hash.to(self.original_hash.device)).sum().item()
            self.attack_success = self.output_hamming >= self.hamming_threshold

            ################################# DELTA SCALEDOWN (OPTIONAL) #############################################

            if self.delta_scaledown:
                scale_factors = torch.linspace(0.0, 1.0, steps=50)
                
                for scale in scale_factors:
                    cand_delta = rgb_delta * scale
                    cand_tensor = self.rgb_tensor + cand_delta
                    cand_targ = self.conversion_func(cand_tensor.clone())

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
            self.output_l2 = self.l2_func(self.rgb_tensor, self.output_tensor)
            
            out = self.output_tensor.detach()
            output_image = ToPILImage()(out)
            output_image.save(output_image_path)
            
            self.log(f"Saved attacked image to {output_image_path}")

            self.log(f"Success status: {self.attack_success}")


        def null_guard(input):
            if input is None:
                return "N/A"
            else:
                return input

        out_log = {
            "pre_validation": {
                "success"               : null_guard(self.attack_success),
                "original_hash"         : null_guard(to_hex(self.original_hash)),
                "output_hash"           : null_guard(to_hex(self.output_hash) if self.output_hash is not None else None),
                "hamming_distance"      : null_guard(self.output_hamming),
                "lpips"                 : null_guard(self.output_lpips),
                "l2"                    : null_guard(self.output_l2),
                "num_steps"             : null_guard(ret_set[1]),
                "ideal_scale_factor"    : null_guard(ret_set[3]),
                "ideal_beta"            : null_guard(ret_set[2])
            },
            "post_validation": image_compare(input_image_path, output_image_path, self.lpips_func, self.device, verbose = "off")
        }

        self.log(out_log)
        return out_log
