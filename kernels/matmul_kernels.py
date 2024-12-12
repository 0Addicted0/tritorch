import triton
import triton.language as tl

from .utils import allow_tf32
from .tune_configs import matmul_kernel_configs
from .act_kernels import apply_act_func


@triton.autotune(
    configs=matmul_kernel_configs(),
    key=["M", "N", "K"],
)
@triton.heuristics({"tf32": lambda _: allow_tf32()})
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    v_ptr,
    pre_act_ptr,  # axb+v
    out_ptr,  # act_func(axb+v)
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_pre_act_m,
    stride_pre_act_n,
    stride_cm,
    stride_cn,
    # fuse option
    add_bias: tl.constexpr,
    act_param: float,
    act_func: tl.constexpr,
    save_pre_act: tl.constexpr,
    fp16: tl.constexpr,
    tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """
    Kernel for computing the matmul C = A x B + v.
    A has shape (M, K), B has shape (K, N), C has shape (M, N) and v has shape (N,).

    Map program ids `pid` to the block of C it should compute.
    每个block负责计算C的一个BLOCK_SIZE_M*BLOCK_SIZE_N的子矩阵

    This is done in a grouped ordering to promote L2 data reuse.
    See above `L2 Cache Optimizations` section for details.

    每个group
    +------------   N  -------------|
    |--BN --|
    +-------------------------------+  ——   ————
    |       |       |       |       |  B     |
    |       |       |       |       |  M     |
    +-------------------------------+  ——    |
    |       |       |       |       |        GROUP_SIZE_M
    |       |       |       |       |        |
    +-------------------------------+        |
    |       |       |       |       |        |
    |       |       |       |       |        |
    +-------------------------------+      ————
    有GROUP_SIZE_M * num_pid_n个block
    这些block的排列方式是列主序的(L2 Cache Optimizations)
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    """
    num_pid_m就是表示行方向上有多少个block
    num_pid_n就是表示列方向上有多少个block
    """
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N).to(tl.int64)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    """
    处理最后一些行方向上不满GROUP_SIZE_M
    """
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    """
    group内的block划分
    """
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_m_masks = (offs_m < M)[:, None]
    offs_n_masks = (offs_n < N)[None, :]
    """
    a_ptrs是一个BLOCK_SIZE_M*BLOCK_SIZE_K的矩阵,每个元素是一个获取A中该位置数据的指针
    """
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            a_ptrs,
            mask=offs_m_masks & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_n_masks & (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # if fp16:
        #     a = a.to(tl.float16)
        #     b = b.to(tl.float16)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, allow_tf32=tf32)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if add_bias:
        v_ptrs = v_ptr + offs_n
        v = tl.load(v_ptrs)
        accumulator += v

    # You can fuse arbitrary activation functions here
    if act_func is not None:
        if save_pre_act:
            pre_act_ptrs = pre_act_ptr + (
                offs_m[:, None] * stride_pre_act_m + offs_n[None, :] * stride_pre_act_n
            )
            tl.store(pre_act_ptrs, accumulator, mask=offs_m_masks & offs_n_masks)

        accumulator = apply_act_func(
            accumulator, None, None, None, act_param, act_func, False
        )

    if fp16:
        accumulator = accumulator.to(tl.float16)

    out_ptrs = out_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    # Write back the block of the output matrix C with masks.
    tl.store(out_ptrs, accumulator, mask=offs_m_masks & offs_n_masks)


@triton.autotune(
    configs=matmul_kernel_configs(),
    key=["M", "N", "K"],
)
@triton.heuristics({"tf32": lambda _: allow_tf32()})
@triton.jit
def batched_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    pre_act_ptr,
    c_ptr,
    bias_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_a_batch,
    stride_am,
    stride_ak,
    stride_b_batch,
    stride_bk,
    stride_bn,
    stride_pre_act_batch,
    stride_pre_act_m,
    stride_pre_act_n,
    stride_c_batch,
    stride_cm,
    stride_cn,
    stride_bias_batch,
    stride_bias_feat,
    # precision
    bias_dim: tl.constexpr,
    fp16: tl.constexpr,
    tf32: tl.constexpr,
    act_param: tl.constexpr,
    act_func: tl.constexpr,  #
    save_pre_act: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """
    Batched matmul kernel for computing the matmul C = act_func(A x B + v, act_param).

    Note:
        A is a tensor, shaped like [batch_size, M, K]
        B is a tensor, shaped like [batch_size, K, N]
        bias is a tensor, shaped like [batch_size, *], where * is either M or N(controlled by bias_dim)
        pre_act_ptr is a tensor, shaped like [batch_size, M, N], used to store the result before activate function
        c_ptr is a tensor, shaped like [batch_size, M, N], used to store the ultimate result
    """
    # L2 optimization matmul
    """
    Map program ids `pid` to the block of C it should compute.
    每个block负责计算C的一个BLOCK_SIZE_M*BLOCK_SIZE_N的子矩阵

    This is done in a grouped ordering to promote L2 data reuse.
    See above `L2 Cache Optimizations` section for details.

    每个group
    +------------   N  -------------|
    |--BN --|
    +-------------------------------+  ——   ————
    |       |       |       |       |  B     |
    |       |       |       |       |  M     |
    +-------------------------------+  ——    |
    |       |       |       |       |        GROUP_SIZE_M
    |       |       |       |       |        |
    +-------------------------------+        |
    |       |       |       |       |        |
    |       |       |       |       |        |
    +-------------------------------+      ————
    有GROUP_SIZE_M * num_pid_n个block
    这些block的排列方式是列主序的(L2 Cache Optimizations)
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    batch_idx = tl.program_id(1).to(tl.int64)
    """
    num_pid_m就是表示行方向上有多少个block
    num_pid_n就是表示列方向上有多少个block
    """
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N).to(tl.int64)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    """
    处理最后一些行方向上不满GROUP_SIZE_M
    """
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    """
    group内的block划分
    """
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    a_batch_base_offset = batch_idx * stride_a_batch
    b_batch_base_offset = batch_idx * stride_b_batch
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    """
    a_ptrs是一个BLOCK_SIZE_M*BLOCK_SIZE_K的矩阵,每个元素是一个获取A中该位置数据的指针
    """
    a_ptrs = (
        a_ptr
        + a_batch_base_offset
        + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        b_ptr
        + b_batch_base_offset
        + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_n[None, :] & (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # if fp16:
        #     a = a.to(tl.float16)
        #     b = b.to(tl.float16)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, allow_tf32=tf32)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if bias_dim >= 0:
        bias_ptr += stride_bias_batch * batch_idx
        if bias_dim == 0:
            bias_ptrs = bias_ptr + offs_m[:, None] * stride_bias_feat
            bias = tl.load(bias_ptrs, mask=mask_m[:, None])
        else:
            bias_ptrs = bias_ptr + offs_n[None, :] * stride_bias_feat
            bias = tl.load(bias_ptrs, mask=mask_n[None, :])
        accumulator += bias

    ## You can fuse arbitrary activation functions here
    pre_act_batch_base_offset = batch_idx * stride_pre_act_batch
    if act_func is not None:
        if save_pre_act:
            pre_act_ptrs = (
                pre_act_ptr
                + pre_act_batch_base_offset
                + (
                    offs_m[:, None] * stride_pre_act_m
                    + offs_n[None, :] * stride_pre_act_n
                )
            )
            tl.store(
                pre_act_ptrs,
                accumulator,
                mask=mask_m[:, None] & mask_n[None, :],
            )

        accumulator = apply_act_func(
            accumulator, None, None, None, act_param, act_func, False
        )
    if fp16:
        accumulator = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_batch_base_offset = batch_idx * stride_c_batch
    c_ptrs = (
        c_ptr
        + c_batch_base_offset
        + (stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :])
    )
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])
