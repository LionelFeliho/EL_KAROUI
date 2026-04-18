"""
Etude numerique de la vitesse de convergence forte du schema d'Euler hybride.

Strategie (methode de Cameron-Clark classique) : on simule avec un pas fin de reference
h_ref = T / N_ref, puis on "sous-echantillonne" les memes increments browniens pour
obtenir des schemas de pas h_n = T / n, avec n << N_ref. L'erreur forte estimee est :

    err_n(p) = ( E sup_{k} |X^{h_n}_{t_k^n} - X^{h_ref}_{t_k^n}|^p )^{1/p}

et on attend un taux  err_n <= C * h_n^{1/2}  (Thm 5.3 de [BCGP23]) pour le goldfish
eta et l'elephant Z, independamment du parametre de Hurst H.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as _gamma

from .kernels import phi_burst
from .simulation import QuadRoughHestonParams, sigma_quad


def simulate_eta_Z_with_fixed_brownian(
    params: QuadRoughHestonParams,
    T: float,
    n: int,
    dW_ref: np.ndarray,
    N_ref: int,
    scheme_Z: str = "stepwise_kernel",
) -> dict:
    """Simule (eta^h, Z^h) sur une grille de pas n, a partir d'increments browniens fixes
    d'une grille fine de pas N_ref (N_ref doit etre multiple de n).

    dW_ref : (n_paths, N_ref) increments fins.
    Retourne eta, Z sur la grille grossiere (n_paths, n+1).
    """
    assert N_ref % n == 0, "N_ref doit etre multiple de n."
    n_paths, _ = dW_ref.shape
    k_ratio = N_ref // n
    # Agregation des increments fins en increments grossiers.
    dW = dW_ref.reshape(n_paths, n, k_ratio).sum(axis=2)     # (n_paths, n)

    alpha = params.alpha
    h = T / n
    t = np.linspace(0.0, T, n + 1)

    eta = np.zeros((n_paths, n + 1))
    burst = params.eta0 * phi_burst(t, alpha)
    F_drift = np.zeros((n_paths, n))
    F_diff = np.zeros((n_paths, n))
    for k in range(n):
        sig_k = np.sqrt(np.maximum(sigma_quad(eta[:, k], params), 0.0))
        drift_k = params.nu * (params.mu - eta[:, k])
        F_drift[:, k] = drift_k * h
        F_diff[:, k] = params.theta * sig_k * dW[:, k]
        eta[:, k + 1] = burst[k + 1] + (
            eta[:, k] - burst[k] + F_drift[:, k] + F_diff[:, k]
        )

    Z = np.zeros((n_paths, n + 1))
    F_total = F_drift + F_diff

    if scheme_Z == "stepwise_kernel":
        W = np.zeros((n + 1, n))
        for kp1 in range(1, n + 1):
            lags = t[kp1] - t[:kp1]
            W[kp1, :kp1] = np.power(lags, alpha - 1.0) / _gamma(alpha)
        Z[:, 1:] = F_total @ W[1:, :].T
    elif scheme_Z == "semi_integrated":
        w_drift = np.zeros((n + 1, n))
        w_diff = np.zeros((n + 1, n))
        for kp1 in range(1, n + 1):
            a_edges = t[kp1] - t[1:kp1 + 1]
            b_edges = t[kp1] - t[:kp1]
            w_drift[kp1, :kp1] = (
                np.power(b_edges, alpha) - np.power(a_edges, alpha)
            ) / (_gamma(alpha + 1.0) * h)
            w_diff[kp1, :kp1] = np.power(b_edges, alpha - 1.0) / _gamma(alpha)
        Z[:, 1:] = F_drift @ w_drift[1:, :].T + F_diff @ w_diff[1:, :].T
    else:
        raise ValueError(scheme_Z)

    return dict(t=t, eta=eta, Z=Z, dW=dW)


def strong_error_study(
    params: QuadRoughHestonParams,
    T: float,
    n_list: list[int],
    N_ref: int,
    n_paths: int = 2000,
    p: float = 2.0,
    rng: np.random.Generator | None = None,
    scheme_Z: str = "stepwise_kernel",
) -> dict:
    """Calcule l'erreur forte sup-norm pour eta et Z pour differentes tailles de grille.

    Returns
    -------
    dict avec :
        n_list
        h_list = T / n_list
        err_eta : (len(n_list),) erreur forte
        err_Z   : (len(n_list),)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    h_ref = T / N_ref
    dW_ref = rng.standard_normal(size=(n_paths, N_ref)) * np.sqrt(h_ref)

    # Reference
    ref = simulate_eta_Z_with_fixed_brownian(
        params, T, N_ref, dW_ref, N_ref, scheme_Z=scheme_Z
    )
    t_ref, eta_ref, Z_ref = ref["t"], ref["eta"], ref["Z"]

    err_eta = np.zeros(len(n_list))
    err_Z = np.zeros(len(n_list))
    for i, n in enumerate(n_list):
        sim = simulate_eta_Z_with_fixed_brownian(
            params, T, n, dW_ref, N_ref, scheme_Z=scheme_Z
        )
        idx = np.linspace(0, N_ref, n + 1, dtype=int)
        diff_eta = sim["eta"] - eta_ref[:, idx]
        diff_Z = sim["Z"] - Z_ref[:, idx]
        sup_eta = np.max(np.abs(diff_eta), axis=1)
        sup_Z = np.max(np.abs(diff_Z), axis=1)
        err_eta[i] = (np.mean(sup_eta ** p)) ** (1.0 / p)
        err_Z[i] = (np.mean(sup_Z ** p)) ** (1.0 / p)

    return dict(
        n_list=np.asarray(n_list),
        h_list=T / np.asarray(n_list),
        err_eta=err_eta,
        err_Z=err_Z,
    )
