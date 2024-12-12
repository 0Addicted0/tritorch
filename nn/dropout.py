"""
Dropout layer with PyTorch autodiff support
"""

import warnings
from random import randint
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch import nn
from torch.amp import custom_fwd, custom_bwd
from triton import cdiv

from kernels.dropout_kernels import dropout_forward_kernel, dropout_backward_kernel
from .types import Context


class DropoutAutoGrad(torch.autograd.Function):
    """
    Differentiable dropout layer.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx: Context, input: Tensor, drop_p: float, training: bool) -> Tensor:
        """
        Randomly zeroes elements in the input.

        Args:
            ctx: Context for variable storage.
            input: Input to perform dropout on. Can have arbitrary shape.
            drop_p: Probability of dropping an element.
            training: Flag indicating if the model is in training mode.

        Returns:
            Input with some elements zeroed out.
        """
        ctx.do_dropout = True

        # if not training or drop_p == 0, no dropout is applied
        if not training or drop_p == 0:
            ctx.do_dropout = False
            return input

        ctx.drop_all = False
        if drop_p == 1:
            ctx.drop_all = True
            return torch.zeros_like(input)

        flattened_input = input.flatten()
        size = flattened_input.shape[0]
        output = torch.empty_like(flattened_input)

        seed = randint(0, 65535)
        ctx.seed = seed
        ctx.drop_p = drop_p

        dropout_forward_kernel[1, 256](flattened_input, output, size, drop_p, seed)
        return output.reshape(input.shape)

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Optional[Tensor], None, None]:
        if not ctx.do_dropout:
            return output_grad, None, None
        if ctx.drop_all:
            return torch.zeros_like(output_grad), None, None

        origin_shape = output_grad.shape
        output_grad = output_grad.flatten()
        size = output_grad.shape[0]
        input_grad = torch.empty_like(output_grad)

        grid = lambda META: (cdiv(size, META["BLOCK_SIZE"]),)
        dropout_backward_kernel[grid](
            output_grad, input_grad, size, ctx.drop_p, ctx.seed
        )
        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(origin_shape), None, None


class Dropout(nn.Dropout):
    """
    Randomly zeroes elements in the input during training.
    See also base class.

    Args:
        p: Probability of dropping an element.
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
    """

    def __init__(self, drop_p: float, inplace: bool = False) -> None:
        super().__init__(p=drop_p, inplace=False)
        self.drop_p = drop_p
        if inplace is True:
            warnings.warn("In-place dropout is not supported out-of-place will be used.")

    def forward(self, input: Tensor) -> Tensor:
        return DropoutAutoGrad.apply(input, self.drop_p, self.training)
