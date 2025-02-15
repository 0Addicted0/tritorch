import triton
import triton.language as tl


@triton.jit
def program_id(
    axis: int,
) -> tl.tensor:
    return tl.program_id(axis).to(tl.int64)


@triton.jit
def num_programs(
    axis: int,
) -> tl.tensor:
    return tl.num_programs(axis).to(tl.int64)
