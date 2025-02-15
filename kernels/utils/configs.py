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


import torch
import triton


def softmax_heur_tile_k(args):
    MAX_TILE_K = 8192
    NUM_SMS = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    tile_k = 1
    upper_bound = min(args["K"], MAX_TILE_K)
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        num_waves = num_blocks / NUM_SMS
        if (num_waves > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k


def softmax_heur_tile_n_non_inner(args):
    return triton.cdiv(8192, args["TILE_K"])


def softmax_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def softmax_heur_num_warps_non_inner(args):
    tile_size = args["TILE_N"] * args["TILE_K"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def softmax_heur_tile_n_inner(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096


def softmax_heur_num_warps_inner(args):
    tile_size = args["TILE_N"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def softmax_heur_tile_n_bwd_non_inner(args):
    return max(1, 1024 // args["TILE_K"])


def softmax_heru_tile_m(args):
    return max(1, 1024 // args["TILE_N"])


SOFTMAX_HEURISTICS_CONFIGS = {
    "softmax_non_inner": {
        "TILE_K": softmax_heur_tile_k,
        "TILE_N": softmax_heur_tile_n_non_inner,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
        "num_warps": softmax_heur_num_warps_non_inner,
    },
    "softmax_inner": {
        "TILE_N": softmax_heur_tile_n_inner,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
        "num_warps": softmax_heur_num_warps_inner,
    },
    "softmax_backward_non_inner": {
        "TILE_N": softmax_heur_tile_n_bwd_non_inner,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
    },
}


def softmax_heuristic_configs(op_name: str):
    if op_name not in SOFTMAX_HEURISTICS_CONFIGS:
        raise RuntimeError("Unknown kernel has no heuristic config")
    return SOFTMAX_HEURISTICS_CONFIGS[op_name]


SOFTMAX_TUNED_CONFIGS = [
    triton.Config({"TILE_K": 32}),
    triton.Config({"TILE_K": 64}),
    triton.Config({"TILE_K": 128}),
    triton.Config({"TILE_K": 256}),
    triton.Config({"TILE_K": 512}),
    triton.Config({"TILE_K": 1024}),
]


def softmax_tuned_configs(op_name: str):
    if op_name not in SOFTMAX_HEURISTICS_CONFIGS:
        raise RuntimeError("Unknown kernel has no heuristic config")

    return SOFTMAX_TUNED_CONFIGS
