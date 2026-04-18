"""
Pricing Monte-Carlo d'options europeennes + inversion de Black-Scholes pour
obtenir les volatilites implicites (smile).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_call(S0: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Prix Black-Scholes d'un call europeen (sans dividendes)."""
    if sigma <= 0.0 or T <= 0.0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_call(price: float, S0: float, K: float, T: float, r: float = 0.0,
                     tol: float = 1e-8) -> float:
    """Inversion de Black-Scholes par brentq. Retourne np.nan si non inversible."""
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    upper = S0
    if price <= intrinsic + tol or price >= upper - tol:
        return np.nan
    f = lambda s: black_scholes_call(S0, K, T, s, r) - price
    try:
        return brentq(f, 1e-6, 5.0, xtol=tol, maxiter=200)
    except ValueError:
        return np.nan


def mc_european_call(S: np.ndarray, K: float, r: float = 0.0, T: float | None = None
                     ) -> tuple[float, float]:
    """Prix Monte-Carlo + ecart-type. S est la matrice (n_paths, N+1) ; T sert a l'actualisation."""
    if T is None:
        T = 0.0
    payoff = np.maximum(S[:, -1] - K, 0.0)
    price = np.exp(-r * T) * payoff.mean()
    se = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, se


def smile_from_paths(S: np.ndarray, strikes: np.ndarray, T: float, S0: float,
                     r: float = 0.0) -> dict:
    """Calcule (prix, IV) pour une grille de strikes sur les trajectoires S."""
    prices = np.zeros_like(strikes, dtype=float)
    ses = np.zeros_like(strikes, dtype=float)
    ivs = np.zeros_like(strikes, dtype=float)
    for i, K in enumerate(strikes):
        p, s = mc_european_call(S, K, r=r, T=T)
        prices[i] = p
        ses[i] = s
        ivs[i] = implied_vol_call(p, S0, K, T, r=r)
    return dict(strikes=strikes, prices=prices, se=ses, iv=ivs)
