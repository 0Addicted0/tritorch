import triton
import triton.language as tl
from triton import next_power_of_2
from triton.language.extra import libdevice
from .utils.configs import reduce3d_kernel_configs


@triton.autotune(
    configs=reduce3d_kernel_configs(),
    key=["shape1", "shape2"],
)
@triton.jit
def max_kernels(
    input_ptr,
    output_ptr,
    indice_ptr,
    shape1,
    shape2,
    in_stride0,
    in_stride1,
    in_stride2,
    out_stride0,
    out_stride1,
    ind_stride0,
    ind_stride1,
    tracking_indices: tl.constexpr,
    fp16: tl.constexpr,
    ind_i64: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Reduce kernel that computes the maximum value of the input tensor at dim=1.

    Args:
        input_ptr (int): pointer to the input tensor.
        output_ptr (int): pointer to the output tensor.
        shape0 (int): size of the 0 dimension.
        shape1 (int): size of the 1 dimension.
        shape2 (int): size of the 2 dimension.
        stride0 (int): stride of the 0 dimension.
        stride1 (int): stride of the 1 dimension.
        stride2 (int): stride of the 2 dimension.
        reduce_size (int): size of the reduce dimension.
        BLOCK_SIZE (int): block size.

    Note any tensor can be shaped like [d1, d2, d3]
    where d1 is the reduce dimension in this kernel.

    Example:
    >>> x = torch.randn(4, 3, 2) -> [4, 1, 2]
    >>> x = torch.randn(4, 3) -> [4, 3, 1] -> [4, 1, 1]
    >>> x = torch.randn(4, 3, 2, 1) -> [4, 6, 1] -> [4, 1, 1]
    """
    # 对于三维张量在dim=1上求max,可以将第0维看成Batch,剩下两维看成一个矩阵
    # 对于每个矩阵,在第0维度上求max相当于每列选一个最大值

    # 一个pid负责reduce一个batch内的所有BLOCK_SIZE列
    pid = tl.program_id(0)
    # 计算当前pid对应的batch和列
    num_pid_col = tl.cdiv(shape2, BLOCK_SIZE_COL)
    batch_idx = pid // num_pid_col
    col_pid_idx = pid % num_pid_col
    # offset
    input_ptr += batch_idx * in_stride0
    col_offs = col_pid_idx * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    col_masks = col_offs < shape2
    accumulator = tl.full((BLOCK_SIZE_COL,), -float("inf"), dtype=tl.float32)
    if tracking_indices:
        accumulator_ind = tl.zeros(
            (BLOCK_SIZE_COL,), dtype=tl.int32 if not ind_i64 else tl.int64
        )
    for row_idx in range(0, shape1, BLOCK_SIZE_ROW):
        # load block
        row_offs = row_idx + tl.arange(0, BLOCK_SIZE_ROW)
        row_masks = row_offs < shape1
        input_ptrs = (
            input_ptr + row_offs[:, None] * in_stride1 + col_offs[None, :] * in_stride2
        )
        blk_max = tl.load(
            input_ptrs,
            mask=row_masks[:, None] & col_masks[None, :],
            other=-float("inf"),
        )
        blk_max, blk_max_ind = tl.max(blk_max, axis=0, return_indices=True)
        # update column-wise max of current block into global accumulator
        if tracking_indices:
            accumulator_ind = tl.where(
                blk_max > accumulator, blk_max_ind + row_idx, accumulator_ind
            )
        accumulator = tl.where(blk_max > accumulator, blk_max, accumulator)
    ## store
    output_ptrs = output_ptr + batch_idx * out_stride0 + col_offs * out_stride1
    if fp16:
        accumulator = accumulator.to(dtype=tl.float16)
    tl.store(
        output_ptrs,
        accumulator,
        mask=tl.arange(0, BLOCK_SIZE_COL) < shape2 - col_pid_idx * BLOCK_SIZE_COL,
    )
    if tracking_indices:
        indice_ptrs = indice_ptr + batch_idx * ind_stride0 + col_offs * ind_stride1
        tl.store(
            indice_ptrs,
            accumulator_ind,
            mask=tl.arange(0, BLOCK_SIZE_COL) < shape2 - col_pid_idx * BLOCK_SIZE_COL,
        )


@triton.autotune(
    # configs=reduce3d_kernel_configs(),
    # bugs for some specific num_warps
    # https://github.com/triton-lang/triton/issues/5327
    configs=[
        triton.Config(
            {"BLOCK_SIZE_ROW": 64, "BLOCK_SIZE_COL": 128}, num_warps=4, num_stages=3
        ),
    ],
    key=["shape1", "shape2"],
)
@triton.jit
def sum_kernels(
    input_ptr,
    output_ptr,
    shape1,
    shape2,
    in_stride0,
    in_stride1,
    in_stride2,
    out_stride0,
    out_stride1,
    avg: tl.constexpr,
    fp16: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Reduce kernel that computes the sum value of the input tensor at dim=1.

    Args:
        input_ptr (int): pointer to the input tensor.
        output_ptr (int): pointer to the output tensor.
        shape1 (int): size of the 1 dimension.
        shape2 (int): size of the 2 dimension.
        stride0 (int): stride of the 0 dimension.
        stride1 (int): stride of the 1 dimension.
        stride2 (int): stride of the 2 dimension.
        avg: whether to compute the average.
        reduce_size (int): size of the reduce dimension.
        BLOCK_SIZE (int): block size.

    Note any tensor can be shaped like [d1, d2, d3]
    where d1 is the reduce dimension in this kernel.

    Example:
    >>> x = torch.randn(4, 3, 2) -> [4, 1, 2]
    >>> x = torch.randn(4, 3) -> [4, 3, 1] -> [4, 1, 1]
    >>> x = torch.randn(4, 3, 2, 1) -> [4, 6, 1] -> [4, 1, 1]
    """
    # 对于三维张量在dim=1上求sum,可以将第0维看成Batch,剩下两维看成一个矩阵
    # 对于每个矩阵,在第0维度上求sum相当于每列选一个最大值

    # 一个pid负责reduce一个batch内的所有BLOCK_SIZE列
    pid = tl.program_id(0)
    # 计算当前pid对应的batch和列
    num_pid_col = tl.cdiv(shape2, BLOCK_SIZE_COL)
    batch_idx = pid // num_pid_col
    col_pid_idx = pid % num_pid_col
    # offset
    input_ptr += batch_idx * in_stride0
    col_offs = col_pid_idx * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    col_masks = col_offs < shape2
    accumulator = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.float32)

    for row_idx in range(0, shape1, BLOCK_SIZE_ROW):
        # load bock
        row_offs = row_idx + tl.arange(0, BLOCK_SIZE_ROW)
        row_masks = row_offs < shape1
        input_ptrs = (
            input_ptr + row_offs[:, None] * in_stride1 + col_offs[None, :] * in_stride2
        )
        blk_sum = tl.load(
            input_ptrs,
            mask=row_masks[:, None] & col_masks[None, :],
            other=0.0,
        ).to(tl.float32)
        # update column-wise sum of current block into global accumulator
        accumulator = accumulator + tl.sum(blk_sum, axis=0)
    if avg:
        accumulator = accumulator / shape1
    ## store
    output_ptrs = output_ptr + batch_idx * out_stride0 + col_offs * out_stride1
    if fp16:
        accumulator = accumulator.to(dtype=tl.float16)
    tl.store(
        output_ptrs,
        accumulator,
        mask=tl.arange(0, BLOCK_SIZE_COL) < shape2 - col_pid_idx * BLOCK_SIZE_COL,
    )


@triton.autotune(configs=[triton.Config({}, num_stages=3, num_warps=4)], key=["nums"])
@triton.heuristics(
    {
        "BLOCK_SIZE": lambda args: next_power_of_2(args["nums"]),
    }
)
@triton.jit
def pnorm_forward_kernels(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    output_stride,
    nums,
    p: tl.constexpr,
    fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implement the element p-norm at the last dim forward kernels.

    Args:
        input_ptr: pointer to the input tensor.
        output_ptr: pointer to the output tensor.
        nums: number of elements to be reduced.
        p: norm type.

    Note:
        input_ptr: pointer to the input tensor, must be a contiguous tensor.
        p: must be positive.
    """
    pid = tl.program_id(0)
    input_ptr += pid * input_stride0
    n_offs = tl.arange(0, BLOCK_SIZE)
    mask = n_offs < nums
    input_ptrs = input_ptr + n_offs * input_stride1
    data = tl.load(input_ptrs, mask=mask, other=0.0)
    data = libdevice.pow(tl.abs(data) + 1e-5, p)
    accmu = tl.sum(data)
    accmu = libdevice.pow(accmu, 1.0 / p)
    if fp16:
        accmu = accmu.to(dtype=tl.float16)
    tl.store(output_ptr + pid * output_stride, accmu)


@triton.autotune(configs=[triton.Config({}, num_stages=3, num_warps=2)], key=["nums"])
@triton.heuristics(
    {
        "BLOCK_SIZE": lambda args: next_power_of_2(args["nums"]),
    }
)
@triton.jit
def pnorm_backward_kernels(
    input_grad_ptr,
    output_grad_ptr,
    output_ptr,
    input_ptr,
    input_grad_stride0,
    input_grad_stride1,
    output_grad_stride,
    output_stride,
    input_stride0,
    input_stride1,
    nums,
    p: tl.constexpr,
    fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implement the element p-norm at the last dim backward kernels.

    dx = dy * y^{1-p} * x * |x|^{p-2}

    Args:
        input_grad_ptr: pointer to the input_grad tensor.
        output_grad_ptr: pointer to the output_grad tensor.
        output_ptr: pointer to the forward output tensor.
        input_ptr: pointer to the input tensor.
        nums: number of elements to be reduced.
        p: norm type.

    Note:
        input_ptr: input tensor, must be a 2D tensor
        p: must be positive.
    """
    pid = tl.program_id(0)
    input_grad_ptr += pid * input_grad_stride0
    output_grad_ptr += pid * output_grad_stride
    output_ptr += pid * output_stride
    input_ptr += pid * input_stride0
    n_offs = tl.arange(0, BLOCK_SIZE)
    mask = n_offs < nums
    input_ptrs = input_ptr + n_offs * input_stride1
    input_data = tl.load(input_ptrs, mask=mask, other=0.0)  # load x
    output_data = tl.load(output_ptr)  # load y
    output_grad_data = tl.load(output_grad_ptr)  # load dy
    # dy * y^{1-p} * x * |x|^{p-2}
    input_grad_data = (
        output_grad_data
        * libdevice.pow(output_data, (1 - p))
        * input_data
        * libdevice.pow(tl.abs(input_data), (p - 2))
    )
    if fp16:
        input_grad_data = input_grad_data.to(dtype=tl.float16)
    input_grad_ptrs = input_grad_ptr + n_offs * input_grad_stride1
    tl.store(input_grad_ptrs, input_grad_data, mask=mask)
