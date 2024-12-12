from typing import Optional
import warnings
import torch
import triton
from torch import Tensor
from kernels.softmax import softmax_warm_kernels, softmax_dim1_warmup
from .types import Context


class SoftmaxAutoGrad(torch.autograd.Function):
    """
    softmax/log_softmax operation with autograd support.

    Note:
        This implementation only supports 2D input tensor on last dim.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, out: Tensor, log: bool) -> Tensor:
        """
        softmax/log_softmax forward operation.

        Args:
            input (Tensor): input tensor.
            dim (int): dimension to reduce.
            out (Tensor): output tensor.
            log (bool): flag indicating if log_softmax was taken.
        """
        n_rows, n_cols = input.shape
        """
        对于nums_programs个block,每个block处理一行
        当数据的rows很多的时候,每个block串行的完成rows/num_programs行
        """
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        fwd_kernel, num_programs = softmax_warm_kernels.get(
            str(BLOCK_SIZE) + str(log) + "fwd", (None, 0)
        )
        if fwd_kernel is None:
            warnings.warn("Softmax kernel function should be warm-up.")
            fwd_kernel, num_programs = softmax_dim1_warmup(input, log, "fwd")
        num_programs = min(num_programs, n_rows)
        output = out if out is not None else torch.empty_like(input)
        # Create a number of persistent programs.
        """
        这里相当于就已经创建了一个实例
        """
        fwd_kernel[(num_programs, 1, 1)](
            input,  # input_ptr
            output,  # output_ptr
            n_rows,  # n_rows
            n_cols,  # n_cols
            input.stride(0),  # input_stride_row
            input.stride(1),  # input_stride_col
            output.stride(0),  # output_stride_row
            output.stride(1),  # output_stride_col
        )
        ctx.log = log
        require_grad = input.requires_grad
        if require_grad:
            ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Optional[Tensor]:
        """
        softmax/log_softmax backward operation.

        Args:
            output_grad (Tensor): output gradients.
        """
        (out,) = ctx.saved_tensors
        n_rows, n_cols = out.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        bwd_kernel, num_programs = softmax_warm_kernels.get(
            str(BLOCK_SIZE) + str(ctx.log) + "bwd", (None, 0)
        )
        if bwd_kernel is None:
            warnings.warn("Softmax kernel function should be warm-up.")
            bwd_kernel, num_programs = softmax_dim1_warmup(out, ctx.log, "bwd")
        num_programs = min(num_programs, n_rows)
        # Create a number of persistent programs.
        input_grad = torch.empty_like(out)
        bwd_kernel[(num_programs, 1, 1)](
            out,  # output_ptr
            output_grad,  # output_grad_ptr
            input_grad,  # input_grad_ptr
            n_rows,  # n_rows
            n_cols,  # n_cols
            output_grad.stride(0),  # output_grad_stride_row
            output_grad.stride(1),  # output_grad_stride_col
            out.stride(0),  # output_stride_row
            out.stride(1),  # output_stride_col
            input_grad.stride(0),  # input_grad_stride_row
            input_grad.stride(1),  # input_grad_stride_col
        )
        return input_grad, None, None, None


def _softmax(
    input: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    log: bool = True,
) -> Tensor:
    """
    softmax reduce operation warpper
    """
    if (dim != -1 and dim != 1) or input.dim() != 2:
        raise RuntimeError(f"Only softmax along the last dimension on 2D is supported.")
    softmax_dim1_warmup(input, log)
    return SoftmaxAutoGrad.apply(input, out, log)
