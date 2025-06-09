import lpips
import numpy as np
import onnxruntime as ort
import os
import torch
from spectra.utils import tensor_resize

class ALEX_ONNX:

    def __init__(self, model_name="lpips_alex.onnx", device="cpu"):
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps" and "MPSExecutionProvider" in ort.get_available_providers():
            providers = ["MPSExecutionProvider", "MPSExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "onnx", model_name)
        self.session = ort.InferenceSession(model_path, providers=providers)


    def get_lpips(self, old_tensor: torch.Tensor, new_tensor: torch.Tensor) -> float:

        old = old_tensor.clone().to(self.device)
        new = new_tensor.clone().to(self.device)

        C, H, W = old.shape

        if H * W < 1024:
            W = 32
            H = 32
            old = tensor_resize(old, H, W)
            new = tensor_resize(new, H, W)

        a3 = torch.zeros_like(old).to(self.device)
        b3 = torch.zeros_like(new).to(self.device)

        if C == 1:
            a = old.view(1, 1, H, W) * 2.0 - 1.0
            b = new.view(1, 1, H, W) * 2.0 - 1.0

            a3 = a.repeat(1, 3, 1, 1)
            b3 = b.repeat(1, 3, 1, 1)

        else:
            a3 = old.unsqueeze(0)
            b3 = new.unsqueeze(0)

        npA = a3.detach().cpu().numpy().astype(np.float32)
        npB = b3.detach().cpu().numpy().astype(np.float32)

        outputs = self.session.run(["lpips_out"], {"inA": npA, "inB": npB})
        return float(outputs[0].item())  # shape (1,1,1,1) -> scalar
    


class ALEX_IMPORT:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = lpips.LPIPS(net='alex').to(device)


    def get_lpips(self, old_tensor: torch.Tensor, new_tensor: torch.Tensor) -> float:
        old = old_tensor.clone().to(self.device)
        new = new_tensor.clone().to(self.device)

        C, H, W = old.shape

        if H * W < 1024:
            W = 32
            H = 32
            old = tensor_resize(old, H, W)
            new = tensor_resize(new, H, W)

        a3 = torch.zeros_like(old).to(self.device)
        b3 = torch.zeros_like(new).to(self.device)

        if C == 1:
            a = old.view(1, 1, H, W) * 2.0 - 1.0
            b = new.view(1, 1, H, W) * 2.0 - 1.0

            a3 = a.repeat(1, 3, 1, 1)
            b3 = b.repeat(1, 3, 1, 1)

        else:
            a3 = old.unsqueeze(0)
            b3 = new.unsqueeze(0)

        output = self.model(a3, b3)
        return output.item()