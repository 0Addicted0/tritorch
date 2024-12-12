import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TRITON_INTERPRET"] = "1"
from typing import Tuple

import torch
from torch import Tensor, autocast
import nn.functional as F

import pytest
from .utils import assert_close, create_input, create_input_like


@pytest.mark.parametrize(
    "input_shape",
    [
        (97, 101),
        (64, 128, 256),
        (64, 32, 48, 48),
        (128, 256, 64, 16),
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_max(
    input_shape: Tuple[int, ...], dim: int, keepdim: bool, dtype: torch.dtype
) -> None:
    if dim >= len(input_shape):
        return
    tritorch_input = create_input(input_shape, dtype=dtype, seed=0)
    pytorch_input = create_input(input_shape, dtype=dtype, seed=0)

    # forward
    tritorch_out, tritorch_ind = F.max(tritorch_input, dim=dim, keepdim=keepdim)
    pytorch_out, pytorch_ind = torch.max(pytorch_input, dim=dim, keepdim=keepdim)

    assert_close(
        (tritorch_out, pytorch_out),
        (tritorch_ind, pytorch_ind),
        rtol=1e-5,
        atol=1e-5,
    )

    tritorch_dy = create_input(tritorch_out.shape, dtype=dtype, seed=1)
    pytorch_dy = create_input(tritorch_out.shape, dtype=dtype, seed=1)

    # backward
    pytorch_out.backward(pytorch_dy)
    tritorch_out.backward(tritorch_dy)

    assert_close((tritorch_input.grad, pytorch_input.grad), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1025,),
        (97, 101),
        (64, 128, 256),
        (64, 32, 48, 48),
        (128, 256, 64, 16),
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("keepdim", [True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sum(
    input_shape: Tuple[int, ...], dim: int, keepdim: bool, dtype: torch.dtype
) -> None:
    if dim >= len(input_shape):
        return
    tritorch_input = create_input(input_shape, dtype=dtype, seed=0)
    pytorch_input = create_input(input_shape, dtype=dtype, seed=0)

    # forward
    tritorch_out = F.sum(tritorch_input, dim=dim, keepdim=keepdim)
    pytorch_out = torch.sum(pytorch_input, dim=dim, keepdim=keepdim)

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-2

    assert_close(
        (tritorch_out, pytorch_out),
        atol=atol,
        rtol=rtol,
    )
    tritorch_dy = create_input(tritorch_out.shape, dtype=dtype, seed=1)
    pytorch_dy = create_input(tritorch_out.shape, dtype=dtype, seed=1)

    # # backward
    pytorch_out.backward(pytorch_dy)
    tritorch_out.backward(tritorch_dy)

    assert_close((tritorch_input.grad, pytorch_input.grad), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1025,),
        (97, 101),
        (64, 128, 256),
        (64, 32, 48, 48),
        (128, 256, 64, 16),
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mean(
    input_shape: Tuple[int, ...], dim: int, keepdim: bool, dtype: torch.dtype
) -> None:
    if dim >= len(input_shape):
        return
    tritorch_input = create_input(input_shape, dtype=dtype, seed=0)
    pytorch_input = create_input(input_shape, dtype=dtype, seed=0)

    # forward
    tritorch_out = F.mean(tritorch_input, dim=dim, keepdim=keepdim)
    pytorch_out = torch.mean(pytorch_input, dim=dim, keepdim=keepdim)

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-2

    assert_close(
        (tritorch_out, pytorch_out),
        atol=atol,
        rtol=rtol,
    )
    tritorch_dy = create_input(tritorch_out.shape, dtype=dtype, seed=1)
    pytorch_dy = create_input(tritorch_out.shape, dtype=dtype, seed=1)

    ## backward
    pytorch_out.backward(pytorch_dy)
    tritorch_out.backward(tritorch_dy)

    assert_close((tritorch_input.grad, pytorch_input.grad), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "input_shape",
    [
        (97, 101),
        (64, 128, 256),
    ],
)
@pytest.mark.parametrize("p", [2.0])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_norm(
    input_shape: Tuple[int, ...], p, keepdim: bool, dtype: torch.dtype
) -> None:
    tritorch_input = create_input(input_shape, dtype=dtype, seed=0)
    pytorch_input = create_input(input_shape, dtype=dtype, seed=0)

    ## forward
    tritorch_out = F.norm(tritorch_input, p=p, dim=-1, keepdim=keepdim, dtype=dtype)
    pytorch_out = torch.norm(pytorch_input, p=p, dim=-1, keepdim=keepdim, dtype=dtype)

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-2

    assert_close(
        (tritorch_out, pytorch_out),
        atol=atol,
        rtol=rtol,
    )

    tritorch_dy = create_input(tritorch_out.shape, dtype=dtype, seed=1)
    pytorch_dy = create_input(tritorch_out.shape, dtype=dtype, seed=1)

    ## backward
    pytorch_out.backward(pytorch_dy)
    tritorch_out.backward(tritorch_dy)

    assert_close((tritorch_input.grad, pytorch_input.grad), atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(2, 2), (128, 256), (99, 107)])
@pytest.mark.parametrize("softmax", ["LogSoftmax"])  # "Softmax",
@pytest.mark.parametrize("input_dtype", [torch.float32])  # , torch.float16])
@pytest.mark.parametrize("amp", [False])  # , True
def test_softmax(
    shape: Tuple[int, ...],
    softmax: str,
    input_dtype: bool,
    amp: bool,
) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    tritorch_input = create_input(shape, dtype=input_dtype)
    pytorch_input = create_input(shape, dtype=input_dtype)

    if softmax == "Softmax":
        with autocast("cuda", enabled=amp):
            tritorch_output = F.softmax(tritorch_input, dim=-1, log=False)
            pytorch_output = torch.nn.functional.softmax(pytorch_input, dim=-1)
    else:
        with autocast("cuda", enabled=amp):
            tritorch_output = F.softmax(tritorch_input, dim=-1, log=True)
            pytorch_output = torch.nn.functional.log_softmax(pytorch_input, dim=-1)

    assert_close((tritorch_output, pytorch_output))

    tritorch_output.backward(create_input_like(tritorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))
    assert_close((tritorch_input.grad, pytorch_input.grad))


@pytest.mark.parametrize(
    "input_shape",
    [
        (64, 256, 256),
        (101, 129, 257),
    ],
)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("amp", [False, True])
def test_nll_loss(
    input_shape: Tuple[int, ...],
    reduction: str,
    weighted: bool,
    input_dtype: bool,
    amp: bool,
) -> None:
    tritorch_input = create_input(input_shape)
    pytorch_input = create_input(input_shape)
    target = torch.randint(
        0, input_shape[1], size=(input_shape[0], input_shape[2]), device="cuda"
    )
    weight = (
        torch.randn(input_shape[1], device="cuda", dtype=torch.float32)
        if weighted
        else None
    )
    with autocast("cuda", enabled=amp):
        tritorch_output = F.nll_loss(
            input=tritorch_input,
            target=target,
            weight=weight,
            reduction=reduction,
            output_dtype=tritorch_input.dtype,
        )
        pytorch_output = torch.nn.functional.nll_loss(
            input=pytorch_input, target=target, weight=weight, reduction=reduction
        )

    atol = 1e-5 if input_dtype == torch.float32 else 1e-2
    rtol = 1e-5 if input_dtype == torch.float32 else 1e-2
    print("tritorch_out:", tritorch_output)
    print("pytorch_out :", pytorch_output)
    assert_close((tritorch_output, pytorch_output), atol=atol, rtol=rtol)

    tritorch_output.backward(create_input_like(tritorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((tritorch_input.grad, pytorch_input.grad), atol=atol, rtol=rtol)
