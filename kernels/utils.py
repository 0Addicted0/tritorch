"""
Utilities for attorch kernels and layers.
"""

from typing import Optional

import torch


def allow_tf32() -> bool:
    """
    Returns whether the current GPU architecture supports TF32.
    """
    return torch.cuda.get_device_capability()[0] >= 8


def get_output_dtype(
    input_dtype: torch.dtype = torch.float32,
    autocast: Optional[str] = None,
) -> torch.dtype:
    """
    Returns the appropriate output dtype for automatic mixed precision
    given the input dtype and the operation's autocast behaviour.

    Args:
        input_dtype: Input dtype.
        autocast: The relevent operation's autocast behaviour.
            None signifies the input dtype should flow through,
            'fp16' signifies autocasting to FP16 when AMP is enabled,
            and 'fp32' signifies autocasting to FP32 when AMP is enabled.
    """
    dtype = torch.get_autocast_dtype("cuda")
    assert dtype, f"Only autocast to float16 is supported, received {dtype}"

    if torch.is_autocast_enabled():
        if autocast is None:
            return input_dtype

        elif autocast == "fp16":
            return torch.float16

        elif autocast == "fp32":
            return torch.float32

        else:
            raise RuntimeError(
                f"Autocast type {autocast} is invalid. "
                "Options are None, fp16, and fp32"
            )

    else:
        return input_dtype
