"""
Common kernel tuning configurations for different kernels.
"""

from typing import List
import triton
import torch


def element_wise_kernel_configs(
    block_name: str = "BLOCK_SIZE",
) -> List[triton.Config]:
    """
    Returns kernel configurations for element-wise operations.

    Args:
        block_name: Name of block argument rows are distributed over.
    """
    return [
        triton.Config({block_name: 64}, num_warps=2),
        triton.Config({block_name: 128}, num_warps=2),
        triton.Config({block_name: 256}, num_warps=4),
        triton.Config({block_name: 512}, num_warps=4),
        triton.Config({block_name: 1024}, num_warps=4),
    ]


def get_n_stages(n_stages: int = 2) -> int:
    """
    Receives number of stages for software pipelining and returns it as-is
    if the GPU architecture is Ampere or newer and 2 otherwise.
    """
    return 2 if torch.cuda.get_device_capability()[0] < 8 else n_stages


def matmul_kernel_configs() -> List[triton.Config]:
    # most frequently used configurations
    # least frequently used
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 1,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 2,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 2,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 4,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ]


def reduce3d_kernel_configs() -> List[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_SIZE_ROW": 256, "BLOCK_SIZE_COL": 256}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_ROW": 256, "BLOCK_SIZE_COL": 128}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_ROW": 256, "BLOCK_SIZE_COL": 128}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_ROW": 128, "BLOCK_SIZE_COL": 128}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_ROW": 64, "BLOCK_SIZE_COL": 128}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_ROW": 64, "BLOCK_SIZE_COL": 64}, num_warps=2, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_ROW": 32, "BLOCK_SIZE_COL": 32}, num_warps=2, num_stages=1
        ),
    ]


def warps_kernel_configs() -> List[triton.Config]:
    """
    Returns kernel configurations with all possible number of warps.
    """
    return [triton.Config({}, num_warps=2)]
    # return [triton.Config({}, num_warps=2**i) for i in range(6)]
