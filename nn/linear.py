"""
Linear layer fused activation.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from triton import cdiv


from kernels.act_kernels import act_func_backward_kernel
from kernels.matmul_kernels import matmul_kernel, batched_gemm_kernel
from .types import Context, Device
from .utils import get_act_func, get_output_dtype


def make_2d_for_mm(input: Tensor) -> Tensor:
    """
    Converts the input to a 2D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 1:
        input = input.unsqueeze(0)
    elif input.ndim >= 3:
        input = input.flatten(0, -2)
    return input


class LinearAutoGrad(torch.autograd.Function):
    """
    Differential for linear layer.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        act_func: Optional[str] = None,
        autocast: str = "fp32",
    ) -> Tensor:
        """
        Linearly transforms the input using weights, optionally adding bias
        and fusing an activation function.

        Args:
            input: Input to transform.
                Must be of shape [..., in_feat_dim].
            weight: Weights input is transformed by.
                Must be of shape [in_feat_dim, out_feat_dim].
            bias: Optional additive bias vector, with None for no bias.
                If provided, must be of shape [out_feat_dim].
            act_func: Name of activation function to apply, with None for identity.
                Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
                'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and
                'leaky_relu_PARAM', where PARAM stands for the parameter in the
                case of parameterized activation functions (e.g., 'leaky_relu_0.01'
                for leaky ReLU with a negative slope of 0.01).

        Returns:
            Input linearly transformed, potentially with added biased and
            fused activation.
        """
        # with param in act_func name (e.g., leaky_relu_0.01)
        act_func, act_param = get_act_func(act_func)

        assert weight.ndim == 2, f"Weights must be 2D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), f"Bias must be 1D, received shape {bias.shape}"

        input_2d = make_2d_for_mm(input)

        assert (
            input_2d.shape[-1] == weight.shape[1]
        ), f"Incompatible input ({input_2d.shape}) and weights ({weight.shape}) shape"
        assert (
            bias is None or weight.shape[0] == bias.shape[0]
        ), f"Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

        M, K = input_2d.shape
        N, _ = weight.shape

        requires_grad = (
            input.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )

        save_pre_act = requires_grad and (act_func is not None)
        output_type = get_output_dtype(input.dtype, autocast=autocast)
        output = torch.empty((M, N), device=input.device, dtype=output_type)
        pre_act = torch.empty_like(output) if save_pre_act else output

        grid = lambda META: (
            cdiv(M, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]),
        )
        matmul_kernel[grid](
            input_2d,
            weight,
            bias,
            pre_act,
            output,
            M,
            N,
            K,
            stride_am=input_2d.stride(0),
            stride_ak=input_2d.stride(1),
            stride_bk=weight.stride(1),  # transpose
            stride_bn=weight.stride(0),
            stride_pre_act_m=pre_act.stride(0),
            stride_pre_act_n=pre_act.stride(1),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            add_bias=bias is not None,
            act_param=act_param,
            act_func=act_func,
            save_pre_act=save_pre_act,
            fp16=output_type is torch.float16,
        )

        ctx.act_param = act_param
        ctx.act_func = act_func
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_type = output_type
        if requires_grad:
            # in `backward`, access by `ctx.saved_tensors`
            ctx.save_for_backward(input, pre_act if save_pre_act else None, weight)

        return output.view(*input.shape[:-1], N)

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of the linear layer.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the linear layer.
        """
        input, pre_act, weight = ctx.saved_tensors

        output_grad_2d = make_2d_for_mm(output_grad)  # [M,N]
        input_2d = make_2d_for_mm(input)

        M, K = input_2d.shape
        N, _ = weight.shape

        assert (
            output_grad_2d.shape[0] == input_2d.shape[0]
            and output_grad_2d.shape[1] == weight.shape[0]
        ), f"Incompatible output gradient ({output_grad_2d.shape}), input ({input_2d.shape}) shape and weights ({weight.shape}) shape"

        if ctx.act_func is None:
            pre_act_grad = output_grad_2d
        else:
            size = M * N
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=pre_act.device)
            grid = lambda META: (cdiv(size, META["BLOCK_SIZE"]),)
            act_func_backward_kernel[grid](
                output_grad,
                pre_act.view_as(pre_act_grad),
                pre_act_grad,
                size,
                None,
                None,
                ctx.act_param,
                ctx.act_func,
                False,
            )
            pre_act_grad = pre_act_grad.view_as(output_grad_2d)
        # dL/db = dL/dy
        bias_grad = pre_act_grad.sum(dim=0) if ctx.bias_requires_grad else None
        if input.requires_grad:
            # dL/dx = dL/dy x W
            input_grad_2d = torch.empty((M, K), dtype=input.dtype, device=input.device)
            grid = lambda META: (
                cdiv(M, META["BLOCK_SIZE_M"]) * cdiv(K, META["BLOCK_SIZE_N"]),
            )
            matmul_kernel[grid](
                pre_act_grad,
                weight,
                None,
                None,
                input_grad_2d,  # dL/dx
                M,
                K,
                N,
                stride_am=pre_act_grad.stride(0),
                stride_ak=pre_act_grad.stride(1),
                stride_bk=weight.stride(0),
                stride_bn=weight.stride(1),
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_cm=input_grad_2d.stride(0),
                stride_cn=input_grad_2d.stride(1),
                add_bias=False,
                act_param=None,
                act_func=None,
                save_pre_act=False,
                fp16=ctx.output_type is torch.float16,
            )
        else:
            input_grad_2d = None
        # dL/dW =dL/dy^T x X
        if weight.requires_grad:
            weight_grad = torch.empty_like(weight)
            grid = lambda META: (
                cdiv(N, META["BLOCK_SIZE_M"]) * cdiv(K, META["BLOCK_SIZE_N"]),
            )
            matmul_kernel[grid](
                pre_act_grad,
                input,
                None,
                None,
                weight_grad,  # dL/dW
                N,
                K,
                M,
                stride_am=pre_act_grad.stride(1),
                stride_ak=pre_act_grad.stride(0),
                stride_bk=input.stride(0),
                stride_bn=input.stride(1),
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_cm=weight_grad.stride(0),
                stride_cn=weight_grad.stride(1),
                add_bias=False,
                act_param=None,
                act_func=None,
                save_pre_act=False,
                fp16=ctx.output_type is torch.float16,
            )
        else:
            weight_grad = None

        return (
            input_grad_2d.view_as(input) if input_grad_2d is not None else None,
            weight_grad,
            bias_grad,
            None,
            None,
        )


class Linear(nn.Linear):
    """
    Linearly transforms the input using weights, optionally adding bias
    and fusing an activation function.
    See also base class.

    Note: Unlike PyTorch's linear layer, the weight matrix in this module is
    of shape [in_features, out_features] instead of [out_features, in_features].
    This may cause unexpected issues when manipulating the weights (e.g., porting
    parameters, initializing them, and so forth).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Flag for additive bias.
        act_func: Name of activation function to apply, with None for identity.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and
            'leaky_relu_PARAM', where PARAM stands for the parameter in the
            case of parameterized activation functions (e.g., 'leaky_relu_0.01'
            for leaky ReLU with a negative slope of 0.01).
        device: Device to use.
        dtype: Dtype of layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func: Optional[str] = None,
        device: Device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_func = act_func

    def forward(self, input: Tensor) -> Tensor:
        return LinearAutoGrad.apply(input, self.weight, self.bias, self.act_func)


class BatchMatmulAutoGrad(torch.autograd.Function):
    """
    Differential for batch matmul.
    """

    @staticmethod
    def forward(
        ctx: Context,
        a: Tensor,
        b: Tensor,
        transpose_a: bool = False,
        transpose_b: bool = False,
        # transpose_c: bool = False,
        act_func: Optional[str] = None,
    ) -> Tensor:
        """
        Batch matrix multiplication, optionally fusing an activation function, transpose A or B.

        Note: a,b must be 3D tensors of shape [batch, ...], [batch, ...].

        Args:
            a: First input tensor.
            b: Second input tensor.
            transpose_a: Flag to transpose A. To keep dim context, output will be transposed too.
            transpose_b: Flag to transpose B.
            act_func: Name of activation function to apply, with None for identity.
                Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
                'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and
                'leaky_relu_PARAM', where PARAM stands for the parameter in the
                case of parameterized activation functions (e.g., 'leaky_relu_0.01'
                for leaky ReLU with a negative slope of 0.01).

        Examples:
        >>> a = torch.randn((2, 2, 4), requires_grad=True)
        >>> b = torch.randn((2, 4, 2), requires_grad=True)
        >>> c = BatchMatmulAutoGrad()(a, b) -> torch.Size([2, 2, 2])

        >>> a = torch.randn((2, 2, 4), requires_grad=True)
        >>> b = torch.randn((2, 2, 4), requires_grad=True)
        >>> c = BatchMatmulAutoGrad()(a, b, transpose_b=True) -> torch.Size([2, 2, 2])

        >>> a = torch.randn((2, 2, 4), requires_grad=True)
        >>> b = torch.randn((2, 2, 4), requires_grad=True)
        >>> c = BatchMatmulAutoGrad()(a, b, transpose_a=True) -> torch.Size([2, 4, 4])
        """
        B, M, Ka = a.shape
        if transpose_a:
            Ka, M = M, Ka
            transpose_c = True
        _, Kb, N = b.shape
        if transpose_b:
            N, Kb = Kb, N
        assert (
            Ka == Kb
            and a.shape[0] == b.shape[0]
            and f"Incompatible input ({a.shape}) and weights ({b.shape}) shape"
        )

        # if transpose_c:
        #     M, N = N, M

        grid = lambda META: (
            cdiv(M, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]),
            B,
        )
        act_func, act_param = get_act_func(act_func)
        save_pre_act = act_func is not None
        pre_act = (
            torch.empty((B, M, N), device=a.device, dtype=a.dtype)
            if save_pre_act
            else None
        )
        c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
        batched_gemm_kernel[grid](
            # Pointers to matrices
            a_ptr=a,
            b_ptr=b,
            pre_act_ptr=pre_act,
            c_ptr=c,
            bias_ptr=None,
            # Matrix dimensions
            M=M,
            N=N,
            K=Ka,
            stride_a_batch=a.stride(0),
            stride_am=a.stride(1) if not transpose_a else a.stride(2),
            stride_ak=a.stride(2) if not transpose_a else a.stride(1),
            stride_b_batch=b.stride(0),
            stride_bk=b.stride(1) if not transpose_b else b.stride(2),
            stride_bn=b.stride(2) if not transpose_b else b.stride(1),
            stride_pre_act_batch=pre_act.stride(0) if save_pre_act else 0,
            stride_pre_act_m=pre_act.stride(1) if pre_act is not None else 0,
            stride_pre_act_n=pre_act.stride(2) if pre_act is not None else 0,
            stride_c_batch=c.stride(0),
            stride_cm=c.stride(1),
            stride_cn=c.stride(2),
            stride_bias_batch=0,
            stride_bias_feat=0,
            # precision
            bias_dim=-1,
            fp16=a.dtype is torch.float16,
            act_param=act_param,
            act_func=act_func,
            save_pre_act=save_pre_act,
        )
        ctx.act_param = act_param
        ctx.act_func = act_func
        ctx.output_type = a.dtype
        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b
        requires_grad = a.requires_grad or b.requires_grad
        if requires_grad:
            # in `backward`, access by `ctx.saved_tensors`
            ctx.save_for_backward(a, b, pre_act if save_pre_act else None)
        return c

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass for batch matmul.
        """
        B, M, N = output_grad.shape
        a, b, pre_act = ctx.saved_tensors
        if ctx.act_func is None:
            pre_act_grad = output_grad
        else:
            size = B * M * N
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=pre_act.device)
            grid = lambda META: (cdiv(size, META["BLOCK_SIZE"]),)
            act_func_backward_kernel[grid](
                output_grad.flatten(),
                pre_act.view_as(pre_act_grad),
                pre_act_grad,
                size,
                None,
                None,
                ctx.act_param,
                ctx.act_func,
                False,
            )
            pre_act_grad = pre_act_grad.view_as(output_grad)

        grad_a = torch.empty_like(a, device=a.device, dtype=a.dtype)
        grad_b = torch.empty_like(b, device=b.device, dtype=b.dtype)

        grid_grad_a = lambda META: (
            cdiv(a.shape[1], META["BLOCK_SIZE_M"])
            * cdiv(a.shape[2], META["BLOCK_SIZE_N"]),
            B,
        )
        grid_grad_b = lambda META: (
            cdiv(b.shape[1], META["BLOCK_SIZE_M"])
            * cdiv(b.shape[2], META["BLOCK_SIZE_N"]),
            B,
        )

        if ctx.transpose_a:
            if ctx.transpose_b:
                """
                y = a^T · b^T
                dL/da = b^T·(dL/dy)^T
                dL/db = (dL/dy)^T·a^T
                """
                batched_gemm_kernel[grid_grad_a](
                    # Pointers to matrices
                    a_ptr=b,
                    b_ptr=pre_act_grad,
                    pre_act_ptr=None,
                    c_ptr=grad_a,  # [B,K,M]
                    bias_ptr=None,
                    # Matrix dimensions
                    M=a.shape[1],  # K
                    N=a.shape[2],  # M
                    K=b.shape[1],  # N
                    stride_a_batch=b.stride(0),
                    stride_am=b.stride(2),
                    stride_ak=b.stride(1),
                    stride_b_batch=pre_act_grad.stride(0),
                    stride_bk=pre_act_grad.stride(2),
                    stride_bn=pre_act_grad.stride(1),
                    stride_pre_act_batch=0,
                    stride_pre_act_m=0,
                    stride_pre_act_n=0,
                    stride_c_batch=grad_a.stride(0),
                    stride_cm=grad_a.stride(1),
                    stride_cn=grad_a.stride(2),
                    stride_bias_batch=0,
                    stride_bias_feat=0,
                    # precision
                    bias_dim=-1,
                    fp16=a.dtype is torch.float16,
                    act_param=None,
                    act_func=None,
                    save_pre_act=False,
                )
                batched_gemm_kernel[grid_grad_b](
                    # Pointers to matrices
                    a_ptr=pre_act_grad,
                    b_ptr=a,
                    pre_act_ptr=None,
                    c_ptr=grad_b,  # [B,N,K]
                    bias_ptr=None,
                    # Matrix dimensions
                    M=b.shape[1],  # N
                    N=b.shape[2],  # K
                    K=a.shape[2],  # M
                    stride_a_batch=pre_act_grad.stride(0),
                    stride_am=pre_act_grad.stride(2),
                    stride_ak=pre_act_grad.stride(1),
                    stride_b_batch=a.stride(0),
                    stride_bk=a.stride(2),
                    stride_bn=a.stride(1),
                    stride_pre_act_batch=0,
                    stride_pre_act_m=0,
                    stride_pre_act_n=0,
                    stride_c_batch=grad_b.stride(0),
                    stride_cm=grad_b.stride(1),
                    stride_cn=grad_b.stride(2),
                    stride_bias_batch=0,
                    stride_bias_feat=0,
                    # precision
                    bias_dim=-1,
                    fp16=a.dtype is torch.float16,
                    act_param=None,
                    act_func=None,
                    save_pre_act=False,
                )
            else:
                """
                y = a^T · b
                dL/da = b·(dL/dy)^T
                dL/db = a·dL/dy
                """
                batched_gemm_kernel[grid_grad_a](
                    # Pointers to matrices
                    a_ptr=b,
                    b_ptr=pre_act_grad,
                    pre_act_ptr=None,
                    c_ptr=grad_a,  # [B,K,M]
                    bias_ptr=None,
                    # Matrix dimensions
                    M=a.shape[1],  # K
                    N=a.shape[2],  # M
                    K=b.shape[2],  # N
                    stride_a_batch=b.stride(0),
                    stride_am=b.stride(1),
                    stride_ak=b.stride(2),
                    stride_b_batch=pre_act_grad.stride(0),
                    stride_bk=pre_act_grad.stride(2),
                    stride_bn=pre_act_grad.stride(1),
                    stride_pre_act_batch=0,
                    stride_pre_act_m=0,
                    stride_pre_act_n=0,
                    stride_c_batch=grad_a.stride(0),
                    stride_cm=grad_a.stride(1),
                    stride_cn=grad_a.stride(2),
                    stride_bias_batch=0,
                    stride_bias_feat=0,
                    # precision
                    bias_dim=-1,
                    fp16=a.dtype is torch.float16,
                    act_param=None,
                    act_func=None,
                    save_pre_act=False,
                )
                batched_gemm_kernel[grid_grad_b](
                    # Pointers to matrices
                    a_ptr=a,
                    b_ptr=pre_act_grad,
                    pre_act_ptr=None,
                    c_ptr=grad_b,  # [B,K,N]
                    bias_ptr=None,
                    # Matrix dimensions
                    M=b.shape[1],  # K
                    N=b.shape[2],  # N
                    K=a.shape[2],  # M
                    stride_a_batch=a.stride(0),
                    stride_am=a.stride(1),
                    stride_ak=a.stride(2),
                    stride_b_batch=pre_act_grad.stride(0),
                    stride_bk=pre_act_grad.stride(1),
                    stride_bn=pre_act_grad.stride(2),
                    stride_pre_act_batch=0,
                    stride_pre_act_m=0,
                    stride_pre_act_n=0,
                    stride_c_batch=grad_b.stride(0),
                    stride_cm=grad_b.stride(1),
                    stride_cn=grad_b.stride(2),
                    stride_bias_batch=0,
                    stride_bias_feat=0,
                    # precision
                    bias_dim=-1,
                    fp16=a.dtype is torch.float16,
                    act_param=None,
                    act_func=None,
                    save_pre_act=False,
                )
        elif ctx.transpose_b:
            """
            y = a · b^T
            dL/da = (dL/dy)·b
            dL/db = (dL/dy)^T·a
            """
            batched_gemm_kernel[grid_grad_a](
                # Pointers to matrices
                a_ptr=pre_act_grad,
                b_ptr=b,
                pre_act_ptr=None,
                c_ptr=grad_a,  # [B,M,K]
                bias_ptr=None,
                # Matrix dimensions
                M=a.shape[1],  # M
                N=a.shape[2],  # K
                K=b.shape[1],  # N
                stride_a_batch=pre_act_grad.stride(0),
                stride_am=pre_act_grad.stride(1),
                stride_ak=pre_act_grad.stride(2),
                stride_b_batch=b.stride(0),
                stride_bk=b.stride(1),
                stride_bn=b.stride(2),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_a.stride(0),
                stride_cm=grad_a.stride(1),
                stride_cn=grad_a.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=a.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
            batched_gemm_kernel[grid_grad_b](
                # Pointers to matrices
                a_ptr=pre_act_grad,
                b_ptr=a,
                pre_act_ptr=None,
                c_ptr=grad_b,  # [B,N,K]
                bias_ptr=None,
                # Matrix dimensions
                M=b.shape[1],  # N
                N=b.shape[2],  # K
                K=a.shape[1],  # M
                stride_a_batch=pre_act_grad.stride(0),
                stride_am=pre_act_grad.stride(2),
                stride_ak=pre_act_grad.stride(1),
                stride_b_batch=a.stride(0),
                stride_bk=a.stride(1),
                stride_bn=a.stride(2),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_b.stride(0),
                stride_cm=grad_b.stride(1),
                stride_cn=grad_b.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=a.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
        else:
            """
            y = a · b
            dL/da = (dL/dy)·b^T
            dL/db = a^T·(dL/dy)
            """
            batched_gemm_kernel[grid_grad_a](
                # Pointers to matrices
                a_ptr=pre_act_grad,
                b_ptr=b,
                pre_act_ptr=None,
                c_ptr=grad_a,  # [B,M,K]
                bias_ptr=None,
                # Matrix dimensions
                M=a.shape[1],  # M
                N=a.shape[2],  # K
                K=b.shape[2],  # N
                stride_a_batch=pre_act_grad.stride(0),
                stride_am=pre_act_grad.stride(1),
                stride_ak=pre_act_grad.stride(2),
                stride_b_batch=b.stride(0),
                stride_bk=b.stride(2),
                stride_bn=b.stride(1),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_a.stride(0),
                stride_cm=grad_a.stride(1),
                stride_cn=grad_a.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=a.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
            batched_gemm_kernel[grid_grad_b](
                # Pointers to matrices
                a_ptr=a,
                b_ptr=pre_act_grad,
                pre_act_ptr=None,
                c_ptr=grad_b,  # [B,K,N]
                bias_ptr=None,
                # Matrix dimensions
                M=b.shape[1],  # K
                N=b.shape[2],  # N
                K=a.shape[1],  # M
                stride_a_batch=a.stride(0),
                stride_am=a.stride(2),
                stride_ak=a.stride(1),
                stride_b_batch=pre_act_grad.stride(0),
                stride_bk=pre_act_grad.stride(1),
                stride_bn=pre_act_grad.stride(2),
                stride_pre_act_batch=0,
                stride_pre_act_m=0,
                stride_pre_act_n=0,
                stride_c_batch=grad_b.stride(0),
                stride_cm=grad_b.stride(1),
                stride_cn=grad_b.stride(2),
                stride_bias_batch=0,
                stride_bias_feat=0,
                # precision
                bias_dim=-1,
                fp16=a.dtype is torch.float16,
                act_param=None,
                act_func=None,
                save_pre_act=False,
            )
        return grad_a, grad_b, None, None, None, None


def bmm(
    a: Tensor,
    b: Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    act_func: Optional[str] = None,
) -> Tensor:
    """
    Function warpper of BatchMatmulAutoGrad.
    """
    return BatchMatmulAutoGrad.apply(a, b, transpose_a, transpose_b, act_func)
