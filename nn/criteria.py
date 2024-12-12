"""
Implementation of Loss Criteria operation with autograd support.
"""

from typing import Optional

import torch
from torch import Tensor
from triton import cdiv

from kernels.nll_loss import (
    nll_loss_forward_kernel,
    nll_loss_backward_kernel,
    BLOCK_SIZE_BATCH_heuristic,
)
from .types import Context
from .reduce import _sum


class NLLLossAutoGrad(torch.autograd.Function):
    """
    NLLLoss operation with autograd support.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        target: Tensor,
        reduction: str = "mean",
        weight: Optional[Tensor] = None,
        output_dtype=torch.float32,
    ) -> Tensor:
        """
        Forward pass of nll loss

        Args:
            input (Tensor): input tensor.
            target (Tensor): target tensor.
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            weight (Tensor, Optional): weight tensor.
        Note:
            weight is to implement weighted mean reduction.
        """
        # sanity check
        assert (
            input.ndim >= 2 and input.ndim <= 3 and target.ndim == 2
        ), f"input ({input.shape}) should be 2D or 3D tensor, target ({target.shape}) should be 2D tensor"

        input_3d = input if input.ndim == 3 else input.unsqueeze(-1)

        if weight is not None:
            assert (
                input_3d.shape[1] == weight.shape[0]
            ), f"input_3d ({input_3d.shape}) and weight ({weight.shape}) should have the same feature size"

        assert (
            input_3d.shape[0] == target.shape[0]
        ), f"input_3d ({input_3d.shape}) and target ({target.shape}) should have the same batch size"
        assert (
            input_3d.shape[2] == target.shape[1]
        ), f"input_3d ({input_3d.shape}) and target ({target.shape}) should have the same spatial size"

        assert reduction in [
            "none",
            "mean",
            "sum",
        ], f"reduction should be 'none', 'mean' or 'sum'"
        ## This check is costly
        # assert torch.all(
        #     target < input_3d.shape[1]
        # ), f"All elements in target should be less than the feature size of input_3d ({input_3d.shape[1]})"

        batch_dim, _, spatial_dim = input_3d.shape
        # twice reduce
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic(
            {"batch_dim": batch_dim, "spatial_dim": spatial_dim}
        )
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
        output_dtype = output_dtype
        sum_weight = (
            torch.empty(out_batch_dim, dtype=torch.float32, device=input.device)
            if reduction == "mean" and weight is not None
            else None
        )
        output = (
            torch.empty_like(target, dtype=output_dtype)
            if reduction == "none"
            else torch.empty(out_batch_dim, dtype=output_dtype, device=input.device)
        )
        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        # fisrt reduce to out_batch_dim
        grid = (cdiv(batch_dim, BLOCK_SIZE_BATCH),)
        nll_loss_forward_kernel[grid](
            input_ptr=input,
            target_ptr=target,
            weight_ptr=weight,
            output_ptr=output,
            sum_weight_ptr=sum_weight,
            batch_dim=batch_dim,
            spatial_dim=spatial_dim,
            input_batch_stride=input.stride(0),
            input_feat_stride=input.stride(1),
            input_spatial_stride=input.stride(2),
            target_batch_stride=target.stride(0),
            target_spatial_stride=target.stride(1),
            weight_stride=weight.stride(0) if weight is not None else 0,
            output_batch_stride=output.stride(0) if reduction == "none" else 1,
            output_spatial_stride=output.stride(1) if reduction == "none" else 1,
            sum_weight_stride=sum_weight.stride(0) if sum_weight is not None else 1,
            fp16=output_dtype is torch.float16,
            reduction=reduction,
            weighted=weight is not None,
        )
        if reduction != "none":
            output = _sum(output, 0)  # scalar

            if reduction == "mean" and weight is not None:
                sum_weight = _sum(sum_weight, 0)
                output /= sum_weight

        ctx.sum_weight = sum_weight  # scalar
        ctx.reduction = reduction
        ctx.weight = weight
        ctx.output_dtype = output_dtype
        if input.requires_grad:
            ctx.save_for_backward(input, target)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        """
        Backward pass of nll loss
        """
        sum_weight = ctx.sum_weight
        reduction = ctx.reduction
        weight = ctx.weight
        output_dtype = ctx.output_dtype
        input, target = ctx.saved_tensors
        input_3d = input if input.ndim == 3 else input.unsqueeze(-1)
        batch_dim, _, spatial_dim = input_3d.shape

        grad_input = torch.zeros_like(
            input_3d, dtype=output_dtype, device=grad_output.device
        )
        grad_output = (
            grad_output.view_as(target) if grad_output.ndim > 0 else grad_output
        )
        grid = lambda META: (cdiv(input_3d.shape[0], META["BLOCK_SIZE_BATCH"]),)

        nll_loss_backward_kernel[grid](
            output_grad_ptr=grad_output,
            target_ptr=target,
            weight_ptr=weight,
            sum_weight_ptr=sum_weight,
            input_grad_ptr=grad_input,
            batch_dim=batch_dim,
            spatial_dim=spatial_dim,
            output_grad_batch_stride=(
                grad_output.stride(0) if grad_output.ndim > 0 else 1
            ),
            output_grad_spatial_stride=(
                grad_output.stride(1) if grad_output.ndim > 0 else 1
            ),
            target_batch_stride=target.stride(0),
            target_spatial_stride=target.stride(1),
            weight_stride=weight.stride(0) if weight is not None else 0,
            input_grad_batch_stride=grad_input.stride(0),
            input_grad_feat_stride=grad_input.stride(1),
            input_grad_spatial_stride=grad_input.stride(2),
            fp16=output_dtype is torch.float16,
            reduction=reduction,
            weighted=weight is not None,
        )

        return grad_input.view_as(input), None, None, None, None


def _nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
    output_dtype=torch.float32,
) -> Tensor:
    return NLLLossAutoGrad.apply(input, target, reduction, weight, output_dtype)
