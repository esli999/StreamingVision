"""Fast checks for JAX recompilation pitfalls in tracking."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax import lax


def test_module_level_jit_reuses_same_scan_length() -> None:
    """Module-level jit should not recompile when scan length repeats."""

    @jax.jit
    def step(carry, _):
        x, = carry
        return (x + 1.0,), x

    def run(n: int) -> float:
        t0 = time.perf_counter()
        out = lax.scan(step, (0.0,), None, length=n)
        out[0][0].block_until_ready()
        return time.perf_counter() - t0

    run(48)
    second = run(48)
    assert second < 0.05, f"expected cache hit for repeated n=48, got {second:.4f}s"


def test_nested_jit_new_function_each_call() -> None:
    """Nested @jax.jit defines a new jitted function object every invocation."""

    def make_step():
        @jax.jit
        def step(carry, _):
            x, = carry
            return (x + 1.0,), x

        return step

    def run(n: int) -> float:
        t0 = time.perf_counter()
        out = lax.scan(make_step(), (0.0,), None, length=n)
        out[0][0].block_until_ready()
        return time.perf_counter() - t0

    run(48)
    second = run(48)
    # Nested pattern typically stays >> 0.05s without a stable function identity.
    assert second > 0.02, f"nested jit should not hit fast path; got {second:.4f}s"
