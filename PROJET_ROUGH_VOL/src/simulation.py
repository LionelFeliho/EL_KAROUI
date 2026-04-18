"""
Schemas d'Euler pour le couple (elephant Z, goldfish eta) de Bonesini-Callegaro-Grasselli-Pages.

Modele de reference (Section 5.3.2 de [BCGP23], inspire de [GJR20] quadratic rough Heston) :

    eta_t = eta_0 * t^{1-alpha}/Gamma(2-alpha)
            + int_0^t (mu - eta_s) * nu ds
            + int_0^t theta * sigma(eta_s) dW_s          (goldfish, markovien)

    Z_t  = int_0^t K_alpha(t-s) * [(mu - eta_s) * nu ds + theta * sigma(eta_s) dW_s]   (elephant, non-markovien)

avec   alpha = H + 1/2   (H le parametre de Hurst, H in (0,1/2) pour rough)
       K_alpha(u) = u^{alpha-1} / Gamma(alpha)
       sigma(y) = a*(y - b)^2 + c   (coefficient quadratique type quadratic rough Heston)

Le couple (Z, eta) est le point d'entree du "pont" non-markov/markov : on simule
eta via un Euler standard, puis on reconstruit Z par convolution fractionnaire.

Variante simulee : on peut aussi prendre le drift de la forme b(eta_s) = nu * (mu - eta_s)
et sigma(eta_s) = theta * chi(eta_s) avec chi(y) = sqrt(max(0, a*(y-b)^2 + c)) si l'on veut
interpretation variance (non negatif). Ici on garde sigma(y) telle quelle et on simule
V_t = a*(eta_t - b)^2 + c separement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
from scipy.special import gamma as _gamma

from .kernels import phi_burst, kernel_integral


# -----------------------------------------------------------------------------
# Parametres du modele
# -----------------------------------------------------------------------------
@dataclass
class QuadRoughHestonParams:
    """Parametres du modele quadratic rough Heston type BCGP/GJR."""
    H: float = 0.1           # parametre de Hurst (rough : H < 1/2)
    mu: float = 0.02         # niveau de reversion
    nu: float = 1.2          # vitesse de reversion
    theta: float = 0.3       # volatilite de la memoire
    a: float = 0.384
    b: float = 0.095
    c: float = 0.0025
    eta0: float = 0.02       # valeur initiale du goldfish (non nulle => burst de memoire)

    @property
    def alpha(self) -> float:
        return self.H + 0.5


def sigma_quad(y: np.ndarray, p: QuadRoughHestonParams) -> np.ndarray:
    """sigma(y) = a * (y - b)^2 + c  (>= c > 0). C'est aussi la variance V."""
    return p.a * (y - p.b) ** 2 + p.c


# -----------------------------------------------------------------------------
# Simulation du couple (eta, Z) -- schema d'Euler hybride
# -----------------------------------------------------------------------------
def simulate_elephant_goldfish(
    params: QuadRoughHestonParams,
    T: float,
    N: int,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    scheme: str = "semi_integrated",
    return_increments: bool = False,
) -> dict:
    """Simule les trajectoires couplees (eta_t, Z_t, V_t) sur [0,T].

    Parameters
    ----------
    params : QuadRoughHestonParams
    T      : horizon de temps
    N      : nombre de pas de temps (pas h = T/N)
    n_paths: nombre de trajectoires Monte-Carlo
    scheme : "semi_integrated" (rate 1/2, Thm 5.3)
             "stepwise_kernel" (schema (5.5), noyau fige sur chaque pas)
    return_increments : si True, renvoie aussi dW et les increments F_l

    Returns
    -------
    dict avec cles :
        t    : (N+1,)   grille de temps
        eta  : (n_paths, N+1) trajectoires du goldfish
        Z    : (n_paths, N+1) trajectoires de l'elephant
        V    : (n_paths, N+1) variance a(eta-b)^2 + c
    """
    if rng is None:
        rng = np.random.default_rng()

    alpha = params.alpha
    h = T / N
    t = np.linspace(0.0, T, N + 1)

    # --- Goldfish eta par Euler standard avec burst initial ---
    sqrt_h = np.sqrt(h)
    dW = rng.standard_normal(size=(n_paths, N)) * sqrt_h     # increments browniens
    eta = np.zeros((n_paths, N + 1))
    # terme de burst : eta_0 * t^{1-alpha}/Gamma(2-alpha) evalue sur la grille
    burst = params.eta0 * phi_burst(t, alpha)                # (N+1,)

    # Increments stochastiques F_l = (mu - eta_l)*nu*h + theta*sigma(eta_l)*dW_l
    #   utilises a la fois pour eta et pour Z.
    F_drift = np.zeros((n_paths, N))
    F_diff = np.zeros((n_paths, N))
    # Integration du terme deterministique : eta0*(phi(t_{k+1}) - phi(t_k)) apparait
    # dans l'incrementation de eta via le terme de burst.
    for k in range(N):
        sig_k = np.sqrt(np.maximum(sigma_quad(eta[:, k], params), 0.0))
        drift_k = params.nu * (params.mu - eta[:, k])
        F_drift[:, k] = drift_k * h
        F_diff[:, k] = params.theta * sig_k * dW[:, k]
        eta[:, k + 1] = burst[k + 1] + (
            eta[:, k] - burst[k] + F_drift[:, k] + F_diff[:, k]
        )
    # La recurrence ci-dessus donne : eta_{k+1} - burst_{k+1} = (eta_k - burst_k) + F_k
    # equivalent a eta_{k+1} = burst_{k+1} + sum_{l<=k} F_l. OK (cf. Eq. 4.2 de BCGP).

    # --- Elephant Z par reconstruction fractionnaire ---
    Z = np.zeros((n_paths, N + 1))
    F_total = F_drift + F_diff                               # (n_paths, N)

    if scheme == "stepwise_kernel":
        # Eq. (5.5) : X_{t_{k+1}} = xi_0 + (1/Gamma(alpha)) * sum K(t_{k+1}-t_l)^{alpha-1} * F_l
        # avec K a l'instant t_{k+1}-t_l ou l=0,...,k.
        weights_matrix = np.zeros((N + 1, N))                # W[k+1, l] = K(t_{k+1} - t_l)
        for kp1 in range(1, N + 1):
            # dt_i = t_{kp1} - t_l pour l=0,...,kp1-1
            lags = t[kp1] - t[:kp1]
            weights_matrix[kp1, :kp1] = np.power(lags, alpha - 1.0) / _gamma(alpha)
        # Z_{k+1} = sum_l weights[k+1,l] * F_l
        Z[:, 1:] = F_total @ weights_matrix[1:, :].T

    elif scheme == "semi_integrated":
        # Eq. (4.6) : X_t = xi_0 + int_0^t K(t-s)[b(s,eta_s) ds + sigma(s,eta_s) dW_s]
        # Au point t_{k+1}, l'incrementation sur [t_l, t_{l+1}] du drift est exactement
        # b(t_l,eta_l) * int_{t_l}^{t_{l+1}} K(t_{k+1}-s) ds
        # = b(t_l,eta_l) * [(t_{k+1}-t_l)^alpha - (t_{k+1}-t_{l+1})^alpha]/Gamma(alpha+1)
        # Pour la partie stochastique, on conserve le choix "noyau fige" a t_{k+1}-t_l
        # (c'est le choix naturel pour un schema discret couple avec eta).
        w_drift = np.zeros((N + 1, N))
        w_diff = np.zeros((N + 1, N))
        for kp1 in range(1, N + 1):
            a_edges = t[kp1] - t[1 : kp1 + 1]                # (t_{k+1}-t_{l+1})
            b_edges = t[kp1] - t[:kp1]                       # (t_{k+1}-t_l)
            w_drift[kp1, :kp1] = (
                np.power(b_edges, alpha) - np.power(a_edges, alpha)
            ) / _gamma(alpha + 1.0) / h                      # b est constant => div par h ne marche pas
            # Astuce : F_drift_l = b_l * h, donc w_drift * F_drift_l = w_drift * b_l * h.
            # On veut : b_l * integral = b_l * [(...)^alpha - (...)^alpha]/Gamma(alpha+1).
            # En stockant w_drift = [(...)^alpha - (...)^alpha]/(h*Gamma(alpha+1)), on a
            # w_drift * F_drift = w_drift * b_l * h = b_l * [(...)^alpha-(...)^alpha]/Gamma(alpha+1). OK.
            w_diff[kp1, :kp1] = np.power(b_edges, alpha - 1.0) / _gamma(alpha)
        Z[:, 1:] = F_drift @ w_drift[1:, :].T + F_diff @ w_diff[1:, :].T

    else:
        raise ValueError(f"scheme inconnu : {scheme}")

    V = sigma_quad(eta, params)

    out = dict(t=t, eta=eta, Z=Z, V=V)
    if return_increments:
        out.update(dW=dW, F_drift=F_drift, F_diff=F_diff)
    return out


# -----------------------------------------------------------------------------
# Simulation de l'actif sous-jacent S
# -----------------------------------------------------------------------------
def simulate_asset(
    params: QuadRoughHestonParams,
    T: float,
    N: int,
    n_paths: int = 1,
    S0: float = 1.0,
    rho: float = -1.0,
    rng: np.random.Generator | None = None,
    scheme: str = "semi_integrated",
) -> dict:
    """Simule S_t avec dS_t = S_t * sqrt(V_t) dB_t, V_t = sigma(eta_t).

    Par defaut rho = -1 (modele "pure feedback" de [GJR20] : le Brownien qui guide
    la variance est le meme que celui qui guide le prix -- au signe pres).
    Pour rho entre -1 et 1, on introduit un Brownien B correle : dB = rho dW + sqrt(1-rho^2) dW_perp.
    """
    if rng is None:
        rng = np.random.default_rng()

    sim = simulate_elephant_goldfish(
        params, T, N, n_paths, rng=rng, scheme=scheme, return_increments=True
    )
    t = sim["t"]
    V = sim["V"]
    dW = sim["dW"]
    h = T / N

    # Brownien correlee
    if abs(rho) < 1.0:
        dWp = rng.standard_normal(size=(n_paths, N)) * np.sqrt(h)
        dB = rho * dW + np.sqrt(1.0 - rho**2) * dWp
    else:
        dB = rho * dW

    # dlog(S_t) = -0.5 * V_t dt + sqrt(V_t) dB_t  (sous Q, r=0)
    V_mid = 0.5 * (V[:, :-1] + V[:, 1:])                     # approx au point median
    sqrtV = np.sqrt(np.maximum(V_mid, 0.0))
    log_ret = -0.5 * V_mid * h + sqrtV * dB
    log_S = np.concatenate([np.full((n_paths, 1), np.log(S0)),
                            np.log(S0) + np.cumsum(log_ret, axis=1)], axis=1)
    S = np.exp(log_S)

    return dict(t=t, S=S, V=V, eta=sim["eta"], Z=sim["Z"])


# -----------------------------------------------------------------------------
# Calcul du "VIX" simule (integrale forward de la variance)
# -----------------------------------------------------------------------------
def vix_squared(
    params: QuadRoughHestonParams,
    t_query: float,
    tau: float = 30 / 365.0,
    n_quad: int = 16,
    n_paths_inner: int = 2000,
    n_paths_outer: int = 1,
    sim_pack: dict | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Approxime VIX_t^2 = (1/tau) E[int_t^{t+tau} V_u du | F_t] par MC imbrique simple.

    Pour un projet de recherche serieux, une approche par quantification ou par une
    formule semi-analytique serait preferable (cf. [GJR20], calibration SPX/VIX).
    Ici on se contente d'un proxy par simulation nested (couteux).
    """
    raise NotImplementedError(
        "vix_squared() n'est pas utilise dans le pipeline principal ; voir VIX approx"
        " non-nested dans la notebook."
    )
