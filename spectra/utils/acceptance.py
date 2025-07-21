from typing import Callable, Tuple
import torch


__all__ = [
    'lpips_acceptance',
    'l2_acceptance',
    'latching_acceptance',
    'step_acceptance',
    'dummy_acceptance'
]


def lpips_acceptance(ao) -> Callable[[torch.Tensor, int], Tuple[bool, bool]]:
    """LPIPS-based acceptance closure bound to an `Attack_Object`."""
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.metrics.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.metrics.current_hamming = int((ao.input_tensors.original_hash != ao.metrics.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)

        if ao.config.deltagrad_verbose:
            print(f"LPIPS: {ao.metrics.current_lpips}")
            print(f"L2: {ao.metrics.current_l2}")
            print(f"Hamming: {ao.metrics.current_hamming}")
            print(f"Gate: {ao.gate}")
            print(f"Threshold: {ao.config.hamming_threshold}")

        break_loop, accepted = False, False
        if ao.gate is not None and ao.metrics.current_lpips >= ao.gate:
            break_loop = True
        
        if ao.metrics.current_hamming >= ao.config.hamming_threshold:
            if ao.metrics.current_lpips < ao.metrics.output_lpips:
                ao.metrics.output_lpips   = ao.metrics.current_lpips
                ao.metrics.output_l2      = ao.metrics.current_l2
                ao.metrics.output_hash    = ao.metrics.current_hash
                ao.metrics.output_hamming = ao.metrics.current_hamming
                accepted = True
            break_loop = True
        return break_loop, accepted
    return _fn


def l2_acceptance(ao):
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.metrics.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.metrics.current_hamming = int((ao.input_tensors.original_hash != ao.metrics.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)

        if ao.config.deltagrad_verbose:
            print(f"LPIPS: {ao.metrics.current_lpips}")
            print(f"L2: {ao.metrics.current_l2}")
            print(f"Hamming: {ao.metrics.current_hamming}")
            print(f"Gate: {ao.gate}")
            print(f"Threshold: {ao.config.hamming_threshold}")

        break_loop, accepted = False, False
        if ao.gate is not None and ao.metrics.current_l2 >= ao.gate:
            break_loop = True

        if ao.metrics.current_hamming >= ao.config.hamming_threshold:
            if ao.metrics.current_l2 < ao.metrics.output_l2:
                ao.metrics.output_l2      = ao.metrics.current_l2
                ao.metrics.output_lpips   = ao.metrics.current_lpips
                ao.metrics.output_hash    = ao.metrics.current_hash
                ao.metrics.output_hamming = ao.metrics.current_hamming
                accepted = True
            break_loop = True
        return break_loop, accepted
    return _fn


def latching_acceptance(ao):
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.metrics.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.metrics.current_hamming = int((ao.input_tensors.original_hash != ao.metrics.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)

        if ao.config.deltagrad_verbose:
            print(f"LPIPS: {ao.metrics.current_lpips}")
            print(f"L2: {ao.metrics.current_l2}")
            print(f"Hamming: {ao.metrics.current_hamming}")
            print(f"Gate: {ao.gate}")
            print(f"Threshold: {ao.config.hamming_threshold}")

        if ao.metrics.current_hamming >= ao.config.hamming_threshold:
            return True, True
        return False, False
    return _fn


def step_acceptance(ao):
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.metrics.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.metrics.current_hamming = int((ao.input_tensors.original_hash != ao.metrics.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)

        if ao.config.deltagrad_verbose:
            print(f"LPIPS: {ao.metrics.current_lpips}")
            print(f"L2: {ao.metrics.current_l2}")
            print(f"Hamming: {ao.metrics.current_hamming}")
            print(f"Gate: {ao.gate}")
            print(f"Threshold: {ao.config.hamming_threshold}")

        if ao.metrics.current_hamming >= ao.config.hamming_threshold:
            if step_number < ao.metrics.min_steps:
                ao.metrics.min_steps = step_number
                return True, True
            return True, False
        return False, False
    return _fn


def dummy_acceptance(ao):
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.metrics.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.metrics.current_hamming = int((ao.input_tensors.original_hash != ao.metrics.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)

        if ao.config.deltagrad_verbose:
            print(f"LPIPS: {ao.metrics.current_lpips}")
            print(f"L2: {ao.metrics.current_l2}")
            print(f"Hamming: {ao.metrics.current_hamming}")
            print(f"Gate: {ao.gate}")
            print(f"Threshold: {ao.config.hamming_threshold}")

        return False, True
    return _fn