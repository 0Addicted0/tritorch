# Triton-torch

### Brief

To learn `backward` pass in mechine learning

This repo aims to implement operators in [triton](https://github.com/triton-lang/triton)

### Features
+ slightly faster than pytorch in inference
+ keep the same style with PyTorch
+ fused kernel, such as bias and activate
+ full test by `pytest`

### structure
kernels/ : kernel functions in triton
nn/ : class/function warpers
test/ : test modules

### Supported operators
+ funcitonal
    + batch_norm
    + max
    + mean
    + sum
    + p-norm
    + bmm
+ layer
    + Linear
+ softmax
+ dropout
+ criteria
    + nll_loss

### Reference:
[attorch](https://github.com/BobMcDear/attorch)