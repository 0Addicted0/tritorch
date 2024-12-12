from typing import Optional, Tuple

import pytest
import torch
from torch import autocast, nn
from torch.nn import functional as F, init

from nn.batch_norm import BatchNorm1d, BatchNorm2d
from .utils import assert_close, create_input, create_input_like


@pytest.mark.parametrize(
    "shape", [(8, 16, 32), (64, 128, 256), (8, 16, 32, 32), (64, 128, 16, 16)]
)
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("momentum", [0.1])
@pytest.mark.parametrize("affine", [True])
@pytest.mark.parametrize("track_running_stats", [True])
@pytest.mark.parametrize("add_pre_act", [True])
@pytest.mark.parametrize("act_func", [None, "relu"])
@pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize("amp", [True])
def test_batch_norm_layer(
    shape: Tuple[int, ...],
    eps: float,
    momentum: float,
    affine: bool,
    track_running_stats: bool,
    add_pre_act: bool,
    act_func: Optional[str],
    input_dtype: bool,
    amp: bool,
) -> None:
    if shape[0] == 1 or (input_dtype is torch.float16 and not amp):
        return

    tritorch_bn = BatchNorm2d if len(shape) == 4 else BatchNorm1d
    bn_name = "BatchNorm2d" if len(shape) == 4 else "BatchNorm1d"
    attorch_input = create_input(shape, dtype=input_dtype)
    pytorch_input = create_input(shape, dtype=input_dtype)

    if add_pre_act:
        tritorch_residual = create_input(shape, dtype=input_dtype, seed=1)
        pytorch_residual = create_input(shape, dtype=input_dtype, seed=1)

    else:
        tritorch_residual = pytorch_residual = None

    tritorch_batch_norm = tritorch_bn(
        num_features=shape[1],
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats,
        act_func=(
            act_func + ("_0.01" if "_" in act_func else "")
            if act_func is not None
            else None
        ),
        dtype=input_dtype,
    )
    pytorch_batch_norm = getattr(nn, bn_name)(
        num_features=shape[1],
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats,
        device="cuda",
        dtype=input_dtype,
    )
    pytorch_act = nn.Identity() if act_func is None else getattr(F, act_func)

    if affine:
        torch.manual_seed(0)
        init.normal_(tritorch_batch_norm.weight)
        init.normal_(tritorch_batch_norm.bias)

        torch.manual_seed(0)
        init.normal_(pytorch_batch_norm.weight)
        init.normal_(pytorch_batch_norm.bias)

    with autocast("cuda", enabled=amp):
        if add_pre_act:
            tritorch_output = tritorch_batch_norm(attorch_input, tritorch_residual)
            pytorch_output = pytorch_act(
                pytorch_batch_norm(pytorch_input) + pytorch_residual
            )

        else:
            tritorch_output = tritorch_batch_norm(attorch_input)
            pytorch_output = pytorch_act(pytorch_batch_norm(pytorch_input))

    atol = 1e-3 if input_dtype is torch.float32 else 1e-1
    rtol = 1e-3 if input_dtype is torch.float32 else 1e-1

    assert_close(
        (tritorch_output, pytorch_output),
        (tritorch_batch_norm.running_mean, pytorch_batch_norm.running_mean),
        (tritorch_batch_norm.running_var, pytorch_batch_norm.running_var),
        atol=atol,
        rtol=rtol,
    )
    tritorch_output.backward(create_input_like(tritorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    residual_grad_pair = (
        (tritorch_residual.grad, pytorch_residual.grad) if add_pre_act else (None, None)
    )
    weight_grad_pair = (
        (tritorch_batch_norm.weight.grad, pytorch_batch_norm.weight.grad)
        if affine
        else (None, None)
    )
    bias_grad_pair = (
        (tritorch_batch_norm.bias.grad, pytorch_batch_norm.bias.grad)
        if affine
        else (None, None)
    )
    assert_close(
        (attorch_input.grad, pytorch_input.grad),
        residual_grad_pair,
        weight_grad_pair,
        bias_grad_pair,
        atol=atol,
        rtol=rtol,
    )
