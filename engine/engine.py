import torch
import utils




class Grayscale_Engine():
    def __init__(self, func, tensor, num_perturbations, device):    #device parameter needs to be the same as the tensor and the func's respective devices
        self.func = func
        self.tensor = tensor.clone().squeeze(0)
        self.tensor_size = self.tensor.numel()
        self.num_perturbations = num_perturbations
        self.device = device
        self.gradient = torch.zeros_like(self.tensor)
        assert self.tensor.dim() == 1, f"Expected 1D tensor, got {self.tensor.dim()}D tensor."



    def set_tensor(self, tensor):
        self.tensor = tensor



    def compute_gradient(self, scale_factor, old_hash):
        perturbations = utils.generate_perturbation_vectors_1d(self.num_perturbations, self.tensor_size // 2, self.device)

        batch_pert = perturbations * scale_factor         
        cand_batch = self.tensor + batch_pert #Broadcast add batch_pert  

        new_hashes = torch.tensor([self.func(v) for v in cand_batch], dtype=torch.int32, device=self.device)    #Will run parallel if device is GPU (just make sure func is also assigned to run on GPU)

        orig_hash = torch.tensor(int(old_hash, 16), dtype=torch.int32, device=self.device)
        x = orig_hash ^ new_hashes                        
        delta_ham = x.bit_count().to(cand_batch.dtype)       

        gradient = (delta_ham.unsqueeze(1) * batch_pert).sum(dim=0)  # [D]
        return gradient







class RGB_Engine():
    def __init__(self, func, tensor, num_perturbations, device):
        self.func = func
        self.tensor = tensor.clone().squeeze(0) #Squeeze to handle RGBA format if present
        self.tensor_size = tensor.numel()
        self.height = tensor.size(1)
        self.width = tensor.size(2)
        self.num_perturbations = num_perturbations
        assert self.tensor.dim() == 3, f"Expected 4D tensor, got {self.tensor.dim()}D tensor."