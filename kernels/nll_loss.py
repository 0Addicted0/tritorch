"""
forward/backward kernels for negtive log likelihood loss(NLLLoss)
"""

import triton
import triton.language as tl
from .utils.configs import warps_kernel_configs


def BLOCK_SIZE_BATCH_heuristic(args) -> int:
    """
    Approximates an appropriate batch block size for NLL loss using a heuristic.

    Args:
        args: Arguments to NLL loss kernel.

    Returns:
        Appropriate batch block size.
    """
    block_size_batch = max(1, triton.next_power_of_2(args["batch_dim"] // 2**10))
    block_size_batch = min(block_size_batch, 128)
    return block_size_batch if args["spatial_dim"] < 64 else 1


@triton.autotune(configs=warps_kernel_configs(), key=["batch_dim", "spatial_dim"])
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH_heuristic,
        "BLOCK_SIZE_SPATIAL": lambda args: triton.next_power_of_2(args["spatial_dim"]),
    }
)
@triton.jit
def nll_loss_forward_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    sum_weight_ptr,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    target_batch_stride,
    target_spatial_stride,
    weight_stride,
    output_batch_stride,
    output_spatial_stride,
    sum_weight_stride,
    fp16: tl.constexpr,
    reduction: tl.constexpr,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    """
    Forward kernel for NLL loss

    Note:
        input_ptr must shape like [batch_dim, feat_dim, spatial_dim]
        target_ptr must shape like [batch_dim, spatial_dim]
        weight_ptr must shape like [feat_dim]
        if we reduce all loss in one block, parallelism is not enough,
        so we do only do a part of reduction in this kernel(batch_dim,spatial_dim->batch_dim//BLOCK_SIZE_BATCH,1)
        sum_weight_ptr* can shape like [batch_dim//BLOCK_SIZE_BATCH]
        output_ptr* can shape like [batch_dim, spatial_dim] or [batch_dim//BLOCK_SIZE_BATCH]

    Args:
        input_ptr: pointer to the input tensor.
        target_ptr: pointer to the target tensor.
        weight_ptr: pointer to the weight tensor.
        output_ptr: pointer to the output tensor.
        sum_weight_ptr: pointer to the sum weight tensor.
        batch_dim: number of batches in the input tensor.
        spatial_dim: number of spatial elements in the input tensor.
        input_batch_stride: stride along the batch dimension of the input tensor.
        input_feat_stride: stride along the feature dimension of the input tensor.
        input_spatial_stride: stride along the spatial dimension of the input tensor.
        target_batch_stride: stride along the batch dimension of the target tensor.
        target_spatial_stride: stride along the spatial dimension of the target tensor.
        weight_spatial_stride: stride along the spatial dimension of the weight tensor.
        output_batch_stride: stride along the batch dimension of the output tensor.
        output_spatial_stride: stride along the spatial dimension of the output tensor.
        reduction: reduction method.
        weighted: flag indicating if weight is provided.
        BLOCK_SIZE_BATCH: block size along the batch dimension.
        BLOCK_SIZE_SPATIAL: block size along the spatial dimension.
    """

    # one program processes BLOCK_SIZE_BATCH batches and BLOCK_SIZE_SPATIAL(>=spatial_dim) elements
    batch_pid = tl.program_id(0).to(tl.int64)
    batch_offs = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH).to(
        tl.int64
    )

    batch_masks = batch_offs < batch_dim

    spatial_offs = tl.arange(0, BLOCK_SIZE_SPATIAL).to(
        tl.int64
    )  # next_power_of_2(spatial_dim)

    spatial_masks = spatial_offs < spatial_dim

    target_ptrs = (
        target_ptr
        + batch_offs[:, None] * target_batch_stride
        + spatial_offs[None, :] * target_spatial_stride
    )
    target = tl.load(
        target_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]
    ).to(tl.int64)

    """
    在一个Batch内, input_ptr的排布是每行表示一个特征维度, 每列表示一个样本
    """
    input_ptrs = (
        input_ptr
        + batch_offs[:, None] * input_batch_stride
        + spatial_offs[None, :] * input_spatial_stride
        + target * input_feat_stride  # this actually do gather operation
    )

    input = tl.load(input_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]).to(
        tl.float32
    )

    output = -input  # here is the unreducted negative log likelihood loss
    if weighted:
        weight_ptrs = weight_ptr + target * weight_stride
        weight = tl.load(
            weight_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]
        ).to(tl.float32)
        output *= weight
    if reduction == "none":
        output_ptrs = (
            output_ptr
            + batch_offs[:, None] * output_batch_stride
            + spatial_offs[None, :] * output_spatial_stride
        )
        if fp16:
            output = output.to(tl.float16)
        tl.store(
            output_ptrs, output, mask=batch_masks[:, None] & spatial_masks[None, :]
        )
    else:
        output_sum = tl.sum(output)
        if reduction == "mean":
            if not weighted:
                # weighted mean do not need div total seperately
                output_sum = output_sum / (batch_dim * spatial_dim)
            output_ptrs = output_ptr + batch_pid * output_batch_stride
            if fp16:
                output_sum = output_sum.to(tl.float16)
            tl.store(output_ptrs, output_sum)
            if weighted:
                # partly reduce
                sum_weight_ptrs = sum_weight_ptr + batch_pid * sum_weight_stride
                # store sum of weight to avoid sample nums affect weighted mean
                # e.g. for 2-classes weighted like[5, 1]
                # if in a batch there are 1 sample in class 0 and 5 samples in class 1
                # sum of weight = 5*1+1*5 = 10
                # so for each class, the weight is 5*1/10=0.5 and 1*5/10=0.5
                sum_weight = tl.sum(weight)
                if fp16:
                    sum_weight = sum_weight.to(tl.float16)
                tl.store(sum_weight_ptrs, sum_weight)
        elif reduction == "sum":
            output_ptrs = output_ptr + batch_pid * output_batch_stride  # partly reduce
            if fp16:
                output_sum = output_sum.to(tl.float16)
            tl.store(output_ptrs, output_sum)


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH_heuristic,
        "BLOCK_SIZE_SPATIAL": lambda args: triton.next_power_of_2(args["spatial_dim"]),
    }
)
@triton.jit
def nll_loss_backward_kernel(
    output_grad_ptr,
    target_ptr,
    weight_ptr,
    sum_weight_ptr,
    input_grad_ptr,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_spatial_stride,
    target_batch_stride,
    target_spatial_stride,
    weight_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    fp16: tl.constexpr,
    reduction: tl.constexpr,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    """
    Backward kernel for NLL loss

    Note:
        output_grad_ptr can shape like [batch_dim, spatial_dim](none reduce)/ [1]
        target_ptr must shape like [batch_dim, spatial_dim]
        weight_ptr must shape like [feat_dim]
        input_grad_ptr must shape like [batch_dim, feat_dim, spatial_dim]
        sum_weight must shape like [1]
    Args:
        output_grad_ptr: pointer to the output gradient tensor.
        target_ptr: pointer to the target tensor.
        weight_ptr: pointer to the weight tensor.
        sum_weight_ptr: pointer to the sum weight tensor.
        input_grad_ptr: pointer to the input gradient tensor.
        batch_dim: number of batches in the input tensor.
        spatial_dim: number of spatial elements in the input tensor.
        output_grad_batch_stride: stride along the batch dimension of the output gradient tensor.
        output_grad_spatial_stride: stride along the spatial dimension of the output gradient tensor.
        target_batch_stride: stride along the batch dimension of the target tensor.
        target_spatial_stride: stride along the spatial dimension of the target tensor.
        weight_stride: stride along the spatial dimension of the weight tensor.
        input_grad_batch_stride: stride along the batch dimension of the input gradient tensor.
        input_grad_feat_stride: stride along the feature dimension of the input gradient tensor.
        input_grad_spatial_stride: stride along the spatial dimension of the input gradient tensor.
        reduction: reduction method.
        weighted: flag indicating if weight is provided.
        BLOCK_SIZE_BATCH: block size along the batch dimension.
        BLOCK_SIZE_SPATIAL: block size along the spatial dimension.
    """
    batch_pid = tl.program_id(0)
    batch_offs = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    batch_masks = batch_offs < batch_dim

    spatial_offs = tl.arange(0, BLOCK_SIZE_SPATIAL)
    spatial_masks = spatial_offs < spatial_dim

    output_grad_masks = None
    output_grad_ptrs = output_grad_ptr
    if reduction == "none":
        output_grad_ptrs = (
            output_grad_ptr
            + batch_offs[:, None] * output_grad_batch_stride
            + spatial_offs[None, :] * output_grad_spatial_stride
        )
        output_grad_masks = batch_masks[:, None] & spatial_masks[None, :]

    output_grad = tl.load(output_grad_ptrs, mask=output_grad_masks).to(tl.float32)
    input_grad = -output_grad  # here we get naive nll loss of input_grad

    target_ptrs = (
        target_ptr
        + batch_offs[:, None] * target_batch_stride
        + spatial_offs[None, :] * target_spatial_stride
    )
    target = tl.load(
        target_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]
    ).to(tl.int64)

    if weighted:
        weight_ptrs = weight_ptr + target * weight_stride  # gather operation
        weight = tl.load(
            weight_ptrs, mask=batch_masks[:, None] & spatial_masks[None, :]
        ).to(tl.float32)
        input_grad *= weight
        if reduction == "mean":
            input_grad /= tl.load(sum_weight_ptr).to(tl.float32)
    elif reduction == "mean":
        input_grad /= batch_dim * spatial_dim

    input_grad_ptrs = (
        input_grad_ptr
        + batch_offs[:, None] * input_grad_batch_stride
        + spatial_offs[None, :] * input_grad_spatial_stride
        + target * input_grad_feat_stride
    )  # here actually do scatter operation
    if fp16:
        input_grad = input_grad.to(tl.float16)
    tl.store(
        input_grad_ptrs, input_grad, mask=batch_masks[:, None] & spatial_masks[None, :]
    )  # scatter operation
