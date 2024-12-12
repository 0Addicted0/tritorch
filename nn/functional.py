from typing import Optional, Any

import torch
from torch import Tensor
from .reduce import _max, _sum, _mean, _norm
from .softmax import _softmax
from .criteria import _nll_loss


def max(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    """
    max reduce operation warpper
    """
    return _max(input, dim, keepdim)


def sum(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    """
    max reduce operation warpper
    """
    return _sum(input, dim, keepdim)


def mean(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    """
    max reduce operation warpper
    """
    return _mean(input, dim, keepdim)


def norm(
    input: Tensor,
    p: float | str | None = 2.0,
    dim: Any | None = -1,
    keepdim: bool = False,
    out: Any | None = None,
    dtype: Any | None = torch.float32,
) -> Tensor:
    return _norm(input, p, dim, keepdim, out, dtype)


def softmax(
    input: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    log: bool = True,
) -> Tensor:
    """
    softmax operation warpper
    """
    return _softmax(input, dim, out, log)


def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
    output_dtype=torch.float32,
) -> Tensor:
    """
    negative log likelihood loss operation warpper
    """
    return _nll_loss(input, target, reduction, weight, output_dtype)
