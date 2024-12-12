import torch
import triton
from triton.runtime import driver
import triton.language as tl


@triton.jit
def softmax_dim1_forward_kernel(
    input_ptr,
    output_ptr,
    n_rows: int,
    n_cols: int,
    input_stride_row: int,
    input_stride_col: int,
    output_stride_row: int,
    output_stride_col: int,
    log: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Implementation of logsoftmax along last dim.

    Note: input_ptr must be a 2D tensor.

    Args:
        input_ptr (int): pointer to the input tensor.
        output_ptr (int): pointer to the output tensor.
        n_rows (int): number of rows in the input tensor.
        n_cols (int): number of columns in the input tensor.
        input_stride_row (int): stride along the row dimension of the input tensor.
        input_stride_col (int): stride along the column dimension of the input tensor.
        output_stride_row (int): stride along the row dimension of the output tensor.
        output_stride_col (int): stride along the column dimension of the output tensor.
        log: flag indicating if log_softmax was taken.
        BLOCK_SIZE (int): block size.
        num_stages (int): number of stages.
    """
    row_start = tl.program_id(0).to(tl.int64)
    row_step = tl.num_programs(0).to(tl.int64)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_stride_row
        """
        BLOCK_SIZE是最小的大于等于n_cols的2的幂
        """
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets * input_stride_col
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
        # softmax_out = exp(x-x.max())/sum(exp(x-x.max())) = exp(row_minus_max)/sum(exp(row_minus_max))
        row_minus_max = row - tl.max(row, axis=0)
        denominator = tl.sum(tl.exp(row_minus_max), axis=0)
        if log:
            # logsoftmax_out = (x-x.max())/ln(sum(exp(x-x.max()))) = row_minus_max/ln(sum(exp(row_minus_max)))
            output = row_minus_max - tl.log(denominator + 1e-8)
        else:
            output = tl.exp(row_minus_max) / (denominator + 1e-8)

        output_row_start_ptr = output_ptr + row_idx * output_stride_row
        output_ptrs = output_row_start_ptr + col_offsets * output_stride_col
        tl.store(output_ptrs, output, mask=col_offsets < n_cols)


@triton.jit
def softmax_dim1_backward_kernel(
    output_ptr,
    output_grad_ptr,
    x_grad_ptr,
    n_rows: int,
    n_cols: int,
    output_stride_row: int,
    output_stride_col: int,
    grad_output_stride_row: int,
    grad_output_stride_col: int,
    grad_x_stride_row: int,
    grad_x_stride_col: int,
    log: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Backward kernel function for softmax/log_softmax along the last dimension.

    Note:
        input_ptr must be a 2D tensor.
        softmax: `dx = y * (dy - (dy*y).sum(-1))`
        log_softmax: `dx = dy - dy.sum(-1) * exp(y)`

    Args:
        input_ptr (int): pointer to the input tensor.
        output_ptr (int): pointer to the output tensor.
        n_rows (int): number of rows in the input tensor.
        n_cols (int): number of columns in the input tensor.
        input_stride_row (int): stride along the row dimension of the input tensor.
    """
    row_start = tl.program_id(0).to(tl.int64)
    row_step = tl.num_programs(0).to(tl.int64)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        """
        BLOCK_SIZE是最小的大于等于n_cols的2的幂
        """
        col_offsets = tl.arange(0, BLOCK_SIZE)
        output_ptrs = (
            output_ptr + row_idx * output_stride_row + col_offsets * output_stride_col
        )
        y = tl.load(output_ptrs, mask=col_offsets < n_cols)
        output_grad_ptrs = (
            output_grad_ptr
            + row_idx * grad_output_stride_row
            + col_offsets * grad_output_stride_col
        )
        dy = tl.load(output_grad_ptrs, mask=col_offsets < n_cols)
        if log:
            x_grad = dy - tl.exp(y) * tl.sum(dy)
        else:
            x_grad = y * (dy - tl.sum(dy * y))
        x_grad_ptrs = (
            x_grad_ptr + row_idx * grad_x_stride_row + col_offsets * grad_x_stride_col
        )
        tl.store(x_grad_ptrs, x_grad, mask=col_offsets < n_cols)


softmax_warm_kernels = {}


def softmax_dim1_warmup(x: torch.Tensor, log: bool, d: str = ""):
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)  # 获取设备参数
    NUM_SM = properties["multiprocessor_count"]  # SM数量
    NUM_REGS = properties["max_num_regs"]  # 每个SM上的寄存器数量
    SIZE_SMEM = properties["max_shared_mem"]  # 每个SM上的共享内存大小
    WARP_SIZE = properties["warpSize"]  # 每个warp的线程数
    n_rows, n_cols = x.shape
    global softmax_warm_kernels
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    ky = str(BLOCK_SIZE) + str(log)
    if (ky + "bwd") in softmax_warm_kernels and (ky + "fwd") in softmax_warm_kernels:
        return
    # Allocate output
    y = torch.empty_like(x)
    num_warps = 8
    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    if (ky + "fwd") not in softmax_warm_kernels:
        # pre-compile forward kernel to get register usage and compute thread occupancy.
        fwd_kernel = softmax_dim1_forward_kernel.warmup(
            input_ptr=x,
            output_ptr=y,
            n_rows=n_rows,
            n_cols=n_cols,
            input_stride_row=x.stride(0),
            input_stride_col=x.stride(1),
            output_stride_row=y.stride(0),
            output_stride_col=y.stride(1),
            log=log,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        fwd_kernel._init_handles()
        """
        kernel.n_regs应该是一个线程需要的寄存器数
        """
        n_regs = max(1, fwd_kernel.n_regs)
        size_smem = max(1, fwd_kernel.metadata.shared)
        """
        这里occypancy表示一个处理器上能并行的block数
        """
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        softmax_warm_kernels[ky + "fwd"] = (fwd_kernel, num_programs)
        if d == "fwd":
            return fwd_kernel, num_programs
    if (ky + "bwd") not in softmax_warm_kernels:
        dy = torch.empty_like(y)
        dx = y
        # pre-compile backward kernel to get register usage and compute thread occupancy.
        bwd_kernel = softmax_dim1_backward_kernel.warmup(
            output_ptr=y,
            output_grad_ptr=dy,
            x_grad_ptr=dx,
            n_rows=n_rows,
            n_cols=n_cols,
            output_stride_row=y.stride(0),
            output_stride_col=y.stride(1),
            grad_output_stride_row=dy.stride(0),
            grad_output_stride_col=dy.stride(1),
            grad_x_stride_row=dx.stride(0),
            grad_x_stride_col=dx.stride(1),
            log=log,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        bwd_kernel._init_handles()
        """
        kernel.n_regs应该是一个线程需要的寄存器数
        """
        n_regs = max(1, bwd_kernel.n_regs)
        size_smem = max(1, bwd_kernel.metadata.shared)
        """
        这里occypancy表示一个处理器上能并行的block数
        """
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        softmax_warm_kernels[ky + "bwd"] = (bwd_kernel, num_programs)
        if d == "bwd":
            return bwd_kernel, num_programs
