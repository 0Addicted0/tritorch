"""
Warping reduce operations with autograd support.
"""

from typing import Optional, Tuple, Any
import torch
import numpy as np
from torch import Tensor
from triton import cdiv

from kernels.reduce_kernels import (
    max_kernels,
    sum_kernels,
    pnorm_forward_kernels,
    pnorm_backward_kernels,
)

from .types import Context


def make_3d_tensor_for_reduce(input: Tensor, dim: int) -> Tensor:
    """
    Make a nD tensor to a 3D tensor for the max reduce operation.

    Note: keep dim in the middle.
    """
    if dim < 0:
        dim = input.dim() + dim
    if dim >= input.dim():
        raise ValueError(f"dim={dim} should be less than input.dim() {input.dim()}")
    if input.dim() == 2:
        if dim == 0:
            return input.unsqueeze(0)
        return input.unsqueeze(-1)
    # Get the shape of the input tensor.
    if dim == 0:
        return make_3d_tensor_for_reduce(input.reshape(input.shape[0], -1), dim)
    elif dim == input.dim() - 1:
        return make_3d_tensor_for_reduce(input.reshape(-1, input.shape[-1]), 1)
    shape = (
        np.prod(input.shape[0:dim]),
        input.shape[dim],
        np.prod(input.shape[dim + 1 :]),
    )
    return input.reshape(shape)


class MaxReduceAutoGrad(torch.autograd.Function):
    """
    Differentiable max reduce operation with autograd support.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        dim: int,
        keepdim: bool = False,
    ) -> Tensor:
        """
        Forward pass of the max reduce operation.
        """
        input_3d = make_3d_tensor_for_reduce(input, dim)
        shape_3d = input_3d.shape
        requires_grad = input.requires_grad
        out_2d = torch.empty(
            (shape_3d[0], shape_3d[2]), dtype=input.dtype, device=input.device
        )
        indices_2d = (
            torch.empty_like(out_2d, dtype=torch.int64, device=input.device)
            if requires_grad
            else None
        )
        grid = lambda META: (shape_3d[0] * cdiv(shape_3d[2], META["BLOCK_SIZE_COL"]),)
        max_kernels[grid](
            input_ptr=input_3d,
            output_ptr=out_2d,
            indice_ptr=indices_2d if requires_grad else None,
            shape1=shape_3d[1],
            shape2=shape_3d[2],
            in_stride0=input_3d.stride(0),
            in_stride1=input_3d.stride(1),
            in_stride2=input_3d.stride(2),
            out_stride0=out_2d.stride(0),
            out_stride1=out_2d.stride(1),
            ind_stride0=indices_2d.stride(0) if requires_grad else 0,
            ind_stride1=indices_2d.stride(1) if requires_grad else 0,
            tracking_indices=requires_grad,
            fp16=input.dtype is torch.float16,
            ind_i64=True,
        )
        out_shape = list(input.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = list(input.shape)
        out_2d = out_2d.view(out_shape)
        if requires_grad:
            indices_2d = indices_2d.view(out_shape)
            ctx.save_for_backward(indices_2d)
        return (out_2d, indices_2d)

    @staticmethod
    def backward(
        ctx: Context, output_grad: Tensor, indices_grad: Tensor
    ) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass of the max reduce operation.

        args include indices_grad
        """
        (indices_2d,) = ctx.saved_tensors
        if not ctx.keepdim:
            indices_2d: torch.Tensor = indices_2d.unsqueeze(ctx.dim)
            output_grad = output_grad.unsqueeze(ctx.dim)

        grad_input = torch.zeros(
            ctx.input_shape, dtype=output_grad.dtype, device=output_grad.device
        )
        with torch.no_grad():
            grad_input.scatter_(ctx.dim, indices_2d, output_grad)
        # make grad_input to the same shape as the input, except the dim
        return grad_input, None, None


def _max(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    """
    max reduce operation warpper
    """
    return MaxReduceAutoGrad.apply(input, dim, keepdim)


class SumReduceAutoGrad(torch.autograd.Function):
    """
    Differentiable sum reduce operation with autograd support.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        dim: int,
        keepdim: bool = False,
    ) -> Tensor:
        """
        Forward pass of the sum reduce operation.
        """
        input_3d = make_3d_tensor_for_reduce(input, dim)
        shape_3d = input_3d.shape
        out_2d = torch.zeros(
            (shape_3d[0], shape_3d[2]), dtype=input.dtype, device=input.device
        )
        grid = lambda META: (shape_3d[0] * cdiv(shape_3d[2], META["BLOCK_SIZE_COL"]),)
        sum_kernels[grid](
            input_ptr=input_3d,
            output_ptr=out_2d,
            shape1=shape_3d[1],
            shape2=shape_3d[2],
            in_stride0=input_3d.stride(0),
            in_stride1=input_3d.stride(1),
            in_stride2=input_3d.stride(2),
            out_stride0=out_2d.stride(0),
            out_stride1=out_2d.stride(1),
            avg=False,
            fp16=input.dtype is torch.float16,
        )
        out_shape = list(input.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = list(input.shape)
        out_2d = out_2d.view(out_shape)
        return out_2d

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass of the max reduce operation.

        args include indices_grad
        """
        if not ctx.keepdim:
            output_grad = output_grad.unsqueeze(ctx.dim)

        grad_input = output_grad.expand(ctx.input_shape)
        # make grad_input to the same shape as the input, except the dim
        return grad_input, None, None


def _sum(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    """
    max reduce operation warpper
    """
    return SumReduceAutoGrad.apply(input, dim, keepdim)


class MeanReduceAutoGrad(torch.autograd.Function):
    """
    Differentiable sum reduce operation with autograd support.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        dim: int,
        keepdim: bool = False,
    ) -> Tensor:
        """
        Forward pass of the sum reduce operation.
        """
        input_3d = make_3d_tensor_for_reduce(input, dim)
        shape_3d = input_3d.shape
        out_2d = torch.zeros(
            (shape_3d[0], shape_3d[2]), dtype=input.dtype, device=input.device
        )
        grid = lambda META: (shape_3d[0] * cdiv(shape_3d[2], META["BLOCK_SIZE_COL"]),)
        sum_kernels[grid](
            input_ptr=input_3d,
            output_ptr=out_2d,
            shape1=shape_3d[1],
            shape2=shape_3d[2],
            in_stride0=input_3d.stride(0),
            in_stride1=input_3d.stride(1),
            in_stride2=input_3d.stride(2),
            out_stride0=out_2d.stride(0),
            out_stride1=out_2d.stride(1),
            avg=True,
            fp16=input.dtype is torch.float16,
        )
        out_shape = list(input.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = list(input.shape)
        out_2d = out_2d.view(out_shape)
        return out_2d

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass of the max reduce operation.

        args include indices_grad
        """
        if not ctx.keepdim:
            output_grad = output_grad.unsqueeze(ctx.dim)

        grad_input = output_grad.expand(ctx.input_shape) / ctx.input_shape[ctx.dim]
        # make grad_input to the same shape as the input, except the dim
        return grad_input, None, None


def _mean(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    """
    mean reduce operation warpper
    """
    return MeanReduceAutoGrad.apply(input, dim, keepdim)


class NormReduceAutoGrad(torch.autograd.Function):
    """
    Autograd for norm reduce operaton.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        p: float | str | None = 2.0,
        dim: Any | None = -1,
        keepdim: bool = False,
        out: Any | None = None,
        dtype: Any | None = torch.float32,
    ):
        if type(p) is str:
            raise ValueError(
                f"only `p-norm` is supported now, p should be a float, but got {type(p)}"
            )
        elif p == None:
            p = 2.0
        p = float(p)
        if dim == -1:
            dim = input.ndim - 1
        assert input.ndim - 1 == dim and "only support last dim norm now"
        out_shape = list(input.shape) if keepdim else list(input.shape)
        input_2d = input.flatten(0, -2)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        output = (
            out
            if out is not None
            else torch.empty(input_2d.shape[0], dtype=dtype, device=input.device)
        )
        requires_grad = input.requires_grad
        pnorm_forward_kernels[(input_2d.shape[0],)](
            input_ptr=input_2d,
            output_ptr=output,
            input_stride0=input_2d.stride(0),
            input_stride1=input_2d.stride(1),
            output_stride=output.stride(0),
            nums=input_2d.shape[1],
            p=p,
            fp16=dtype is torch.float16,
        )
        ctx.p = p
        if requires_grad:
            ctx.save_for_backward(input, output)
        return output.reshape(out_shape)

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        y = (\sum |x_i|^p)^{1/p}
        dy/dx_i = (1/p) * (\sum |x_i|^p)^{1/p-1}*p*|x_i|^{p-1}*dsign(x_i)
                = (\sum |x_i|^p)^{1/p-1} * |x_i|^{p-1} * dsign(x_i)
                = (\sum |x_i|^p)^{1/p-1} * x_i * |x_i|^{p-2}
                = y^{1-p} * x_i * |x_i|^{p-2}
        dx = dy * y^{1-p} * x * |x|^{p-2}
        """
        input, output = ctx.saved_tensors
        input_2d: Tensor = input.flatten(0, -2)
        input_grad_2d = torch.empty_like(
            input_2d, dtype=output_grad.dtype, device=output_grad.device
        )
        output_grad_2d = output_grad.flatten()
        p = ctx.p
        pnorm_backward_kernels[(input_2d.shape[0],)](
            input_grad_ptr=input_grad_2d,
            output_grad_ptr=output_grad_2d,
            output_ptr=output,
            input_ptr=input_2d,
            input_grad_stride0=input_grad_2d.stride(0),
            input_grad_stride1=input_grad_2d.stride(1),
            output_grad_stride=output_grad_2d.stride(0),
            output_stride=output.stride(0),
            input_stride0=input_2d.stride(0),
            input_stride1=input_2d.stride(1),
            nums=input_2d.shape[1],
            p=p,
            fp16=output_grad.dtype is torch.float16,
        )
        return input_grad_2d.reshape(input.shape), None, None, None, None, None


def _norm(
    input: Tensor,
    p: float | str | None = 2.0,
    dim: Any | None = -1,
    keepdim: bool = False,
    out: Any | None = None,
    dtype: Any | None = torch.float32,
) -> Tensor:
    """
    p-norm reduce operation warpper
    """
    return NormReduceAutoGrad.apply(input, p, dim, keepdim, out, dtype)
