import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# System / GPU Utilities
# =============================================================================

def check_gpus():
    n = torch.cuda.device_count()
    print(f"Available GPUs: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, Memory: {props.total_memory / (1024**2):.0f} MB")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_non_differential_modules():
    # Dropout introduces stochasticity that corrupts Hessian estimates.
    F.dropout = lambda x, *args, **kwargs: x
    # Flash/memory-efficient attention kernels do not support second-order
    # derivatives required by vhp(); fall back to the math backend.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


# =============================================================================
# Model Utilities
# =============================================================================

def get_all_blocks(model):
    """Return the list of transformer decoder blocks, unwrapping PEFT/HF wrappers as needed."""
    inner = model
    for _ in range(6):
        if hasattr(inner, 'layers') and isinstance(inner.layers, torch.nn.ModuleList):
            return inner.layers
        if hasattr(inner, 'blocks') and isinstance(inner.blocks, torch.nn.ModuleList):
            return inner.blocks
        if hasattr(inner, 'model'):
            inner = inner.model
        elif hasattr(inner, 'base_model'):
            inner = inner.base_model
        else:
            break
    raise ValueError("Cannot locate transformer blocks in model")


def get_nested_attr(obj, attr_path: str):
    """Traverse a dot-separated attribute path, e.g. 'self_attn.q_proj.lora_B'."""
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj


# =============================================================================
# Math Helpers for Lanczos
# =============================================================================

def move_to_device(params: List[torch.Tensor], device: str) -> List[torch.Tensor]:
    return [p.to(device) for p in params]


def group_product(xs, ys) -> torch.Tensor:
    """Compute the dot product between two lists of tensors, summed element-wise."""
    return sum(torch.sum(x * y) for x, y in zip(xs, ys))


def group_add(params, update, alpha=1):
    """Compute params + alpha * update element-wise across two tensor lists."""
    return [p + u * alpha for p, u in zip(params, update)]


def normalization(v):
    """Normalise a list of tensors as if they were a single flattened vector."""
    s = group_product(v, v) ** 0.5
    s = s.cpu().item() if torch.is_tensor(s) else s
    return [vi / (s + 1e-6) for vi in v]