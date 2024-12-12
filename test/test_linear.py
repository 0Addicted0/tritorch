# import os
# os.environ["TRITON_INTERPRET"] = "1"
from typing import Optional, Tuple

import pytest
import torch
from torch import autocast, nn
from nn.linear import Linear, bmm
from torch.nn import functional as F

from .utils import assert_close, create_input, create_input_like


@pytest.mark.parametrize("input_shape", [(128, 257), (512, 1024)])
@pytest.mark.parametrize("out_dim", [196, 384, 512])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize(
    "act_func",
    [
        None,
        # "sigmoid",
        # "tanh",
        "relu",
        # "gelu",
        # "silu",
        # "relu6",
        # "hardsigmoid",
        # "hardswish",
        # "selu",
        # "mish",
        # "leaky_relu",
    ],
)
@pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize("amp", [False, True])
def test_linear(
    input_shape: Tuple[int, ...],
    out_dim: int,
    bias: bool,
    act_func: Optional[str],
    input_dtype: bool,
    amp: bool,
) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    tritorch_input = create_input(input_shape, dtype=input_dtype)
    pytorch_input = create_input(input_shape, dtype=input_dtype)

    torch.manual_seed(0)
    tritorch_linear = Linear(
        input_shape[-1],
        out_dim,
        bias=bias,
        act_func=(
            act_func + ("_0.01" if "_" in act_func else "")
            if act_func is not None
            else None
        ),
        dtype=input_dtype,
    )

    torch.manual_seed(0)
    pytorch_linear = nn.Linear(
        input_shape[-1], out_dim, bias=bias, device="cuda", dtype=input_dtype
    )
    pytorch_act = nn.Identity() if act_func is None else getattr(F, act_func)

    ## forward test
    with autocast("cuda", enabled=amp):
        tritorch_output = tritorch_linear(tritorch_input)
        pytorch_output = pytorch_act(pytorch_linear(pytorch_input))
    atol = 1e-4 if input_dtype == torch.float32 else 1e-1
    rtol = 1e-4 if input_dtype == torch.float32 else 1e-1
    assert_close((tritorch_output, pytorch_output), atol=atol, rtol=rtol)

    ## backward test
    tritorch_output.backward(create_input_like(tritorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close(
        (tritorch_input.grad, pytorch_input.grad),
        (tritorch_linear.weight.grad, pytorch_linear.weight.grad),
        (tritorch_linear.bias.grad, pytorch_linear.bias.grad) if bias else (None, None),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shape", [((79, 129), (129, 256)), ((128, 256), (256, 512))])
# @pytest.mark.parametrize("shape", [[[3, 4], [4, 5]]])
@pytest.mark.parametrize(
    "act_func",
    [
        None,
        # "sigmoid",
        # "tanh",
        # "relu",
        # "gelu",
        # "silu",
        # "relu6",
        # "hardsigmoid",
        # "hardswish",
        # "selu",
        # "mish",
        # "leaky_relu",
    ],
)
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_bmm(
    batch_size: int,
    shape: Tuple[Tuple[int, int]],
    act_func: Optional[str],
    transpose_a: bool,
    transpose_b: bool,
    dtype: torch.dtype,
) -> None:
    a_shape, b_shape = shape
    if transpose_a:
        a_shape = a_shape[::-1]
    if transpose_b:
        b_shape = b_shape[::-1]
    torch.manual_seed(0)
    a = torch.randn(
        (batch_size, *a_shape), dtype=dtype, device="cuda", requires_grad=True
    )
    b = torch.randn(
        (batch_size, *b_shape), dtype=dtype, device="cuda", requires_grad=True
    )
    if transpose_a:
        th_a = a.detach().transpose_(1, 2).clone().requires_grad_()
    else:
        th_a = a.detach().clone().requires_grad_()
    if transpose_b:
        th_b = b.detach().transpose_(1, 2).clone().requires_grad_()
    else:
        th_b = b.detach().clone().requires_grad_()

    atol = 1e-4 if dtype == torch.float32 else 1e-1
    rtol = 1e-4 if dtype == torch.float32 else 1e-1

    # forward_check
    th_bmm = torch.bmm(th_a, th_b)
    pytorch_act = nn.Identity() if act_func is None else getattr(F, act_func)
    th_c: torch.Tensor = pytorch_act(th_bmm)
    c = bmm(a, b, transpose_a, transpose_b, act_func)
    assert_close((c, th_c), rtol=rtol, atol=atol)

    ## backward_check
    torch.manual_seed(1)
    dc = torch.randn_like(c, dtype=dtype, device=c.device)
    torch.manual_seed(1)
    th_dc = torch.randn_like(th_c, dtype=dtype, device="cuda")
    c.backward(dc)
    th_c.backward(th_dc)
    grad_a = a.grad if not transpose_a else a.grad.transpose(1, 2)
    grad_b = b.grad if not transpose_b else b.grad.transpose(1, 2)
    assert_close(
        (grad_a, th_a.grad),
        (grad_b, th_b.grad),
        rtol=rtol,
        atol=atol,
    )
