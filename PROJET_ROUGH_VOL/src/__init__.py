"""Package de simulation pour le modele quadratic rough Heston "elephant/goldfish"."""

from .kernels import (
    fractional_kernel,
    fractional_cokernel,
    phi_burst,
    kernel_integral,
    exponential_kernel,
    gamma_kernel,
)
from .simulation import (
    QuadRoughHestonParams,
    sigma_quad,
    simulate_elephant_goldfish,
    simulate_asset,
)
from .pricing import (
    black_scholes_call,
    implied_vol_call,
    mc_european_call,
    smile_from_paths,
)
from .convergence import strong_error_study

__all__ = [
    "fractional_kernel",
    "fractional_cokernel",
    "phi_burst",
    "kernel_integral",
    "exponential_kernel",
    "gamma_kernel",
    "QuadRoughHestonParams",
    "sigma_quad",
    "simulate_elephant_goldfish",
    "simulate_asset",
    "black_scholes_call",
    "implied_vol_call",
    "mc_european_call",
    "smile_from_paths",
    "strong_error_study",
]
