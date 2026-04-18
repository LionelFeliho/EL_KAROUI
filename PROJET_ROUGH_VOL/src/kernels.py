"""
Noyaux de convolution et co-noyaux pour les equations de Volterra stochastiques.

References
----------
[BCGP23] Bonesini, Callegaro, Grasselli, Pages.
         "From elephant to goldfish (and back): memory in stochastic Volterra processes",
         arXiv:2306.02708v2, 2023.

Notation : on fixe lambda = 0 (cas non amorti) sauf mention contraire.
Pour le noyau fractionnaire K_{1,alpha}(t) = t^{alpha-1}/Gamma(alpha) avec alpha = H + 1/2,
le co-noyau est K-bar_{1,1-alpha}(t) = t^{-alpha}/Gamma(1-alpha).
La convolution (K * K-bar)(t) = 1 (resolvant du premier type).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as _gamma


# -----------------------------------------------------------------------------
# Noyau fractionnaire et co-noyau
# -----------------------------------------------------------------------------
def fractional_kernel(t: np.ndarray, alpha: float) -> np.ndarray:
    """K_{1,alpha}(t) = t^{alpha-1} / Gamma(alpha), t > 0.

    Parameters
    ----------
    t     : array_like, valeurs positives de temps
    alpha : ordre du noyau (alpha = H + 1/2, avec H le parametre de Hurst)
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = np.power(t[pos], alpha - 1.0) / _gamma(alpha)
    return out


def fractional_cokernel(t: np.ndarray, alpha: float) -> np.ndarray:
    """Co-noyau K-bar_{1,1-alpha}(t) = t^{-alpha} / Gamma(1-alpha)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = np.power(t[pos], -alpha) / _gamma(1.0 - alpha)
    return out


def phi_burst(t: np.ndarray, alpha: float) -> np.ndarray:
    """Terme de 'burst' initial : phi(t) = t^{1-alpha} / Gamma(2-alpha).

    C'est (K-bar * 1)(t) dans la dynamique du goldfish eta dans [BCGP23, eq. (5.7)] :
        eta_t = eta_0 * phi(t) + int_0^t b(s,eta_s) ds + int_0^t sigma(s,eta_s) dW_s.
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = np.power(t[pos], 1.0 - alpha) / _gamma(2.0 - alpha)
    return out


# -----------------------------------------------------------------------------
# Integrales semi-analytiques utiles pour le schema pseudo-Euler
# -----------------------------------------------------------------------------
def kernel_integral(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Calcule int_a^b (u)^{alpha-1}/Gamma(alpha) du = [b^alpha - a^alpha]/Gamma(alpha+1).

    Utilise pour integrer exactement le noyau sur un pas de temps dans le schema
    "semi-integrated" (eq. 4.6 de [BCGP23]).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (np.power(b, alpha) - np.power(a, alpha)) / _gamma(alpha + 1.0)


# -----------------------------------------------------------------------------
# Autres noyaux pour reference
# -----------------------------------------------------------------------------
def exponential_kernel(t: np.ndarray, c: float = 1.0, lam: float = 1.0) -> np.ndarray:
    """K_{c,1,lambda}(t) = c * exp(-lambda * t)."""
    t = np.asarray(t, dtype=float)
    return c * np.exp(-lam * t) * (t > 0)


def gamma_kernel(t: np.ndarray, c: float, alpha: float, lam: float) -> np.ndarray:
    """K_{c,alpha,lambda}(t) = c e^{-lambda t} t^{alpha-1}/Gamma(alpha)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = c * np.exp(-lam * t[pos]) * np.power(t[pos], alpha - 1.0) / _gamma(alpha)
    return out
