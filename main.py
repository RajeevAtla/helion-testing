"""
Helion Softmax Kernel Examples
==============================
This example demonstrates multiple Helion kernel implementations of the softmax function,
including a simple wrapper around PyTorch's softmax, and a numerically optimized two-pass version.
The example also includes a check function to compare these kernels against PyTorch's
built-in softmax for correctness.
"""

# %%
from __future__ import annotations
import torch
import helion
from helion._testing import run_example
import helion.language as hl
from helion.autotuner.random_search import RandomSearch
from helion.autotuner.local_cache import LocalAutotuneCache

def random_search_autotuner(kernel, args, **kwargs):
    # kwargs is whatever Helion passes in; you can ignore it or use it
    return LocalAutotuneCache(
        RandomSearch(
            kernel,
            args,
            count=20000,  # number of random configs to try (default is 1000)
        )
    )

@helion.kernel(
    autotune_effort="full",
    autotuner_fn=random_search_autotuner,
    autotune_random_seed=42,
    autotune_compile_timeout=300,
    force_autotune=True,
    static_shapes=True,
    dot_precision="tf32",
    allow_warp_specialize=True,
    persistent_reserved_sms=0,
    autotune_max_generations=500,
    autotune_rebenchmark_threshold=1.05
)
def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Simple Helion kernel wrapping PyTorch's softmax function.
    Args:
        x (torch.Tensor): Input tensor of shape [n, m].
    Returns:
        torch.Tensor: Softmax output tensor of the same shape.
    """
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


# %%
def check(m: int, n: int) -> None:
    """
    Runs correctness checks comparing Helion softmax kernels against PyTorch's softmax.
    Args:
        m (int): Number of rows in input tensor.
        n (int): Number of columns in input tensor.
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    run_example(softmax, lambda x: torch.nn.functional.softmax(x, dim=1), (x,))


# %%
def main() -> None:
    """
    Main function to run the softmax kernel correctness check with example input size.
    """
    check(4096, 4096)


# %%
if __name__ == "__main__":
    main()

