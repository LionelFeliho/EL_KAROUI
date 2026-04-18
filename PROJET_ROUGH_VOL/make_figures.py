"""Genere toutes les figures du rapport.

Execute depuis la racine du projet :
    python make_figures.py
Les figures sont ecrites dans report/figures/.
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import (
    QuadRoughHestonParams,
    fractional_kernel,
    simulate_elephant_goldfish,
    simulate_asset,
    smile_from_paths,
    strong_error_study,
)

# Configuration graphique
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.9,
})

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    path_pdf = os.path.join(FIG_DIR, name + ".pdf")
    path_png = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight")
    print(f"  -> {name}.pdf/.png")
    plt.close(fig)


# ============================================================================
# FIGURE 1 -- Noyau fractionnaire pour differents H
# ============================================================================
def fig_kernel_shapes():
    print("[fig 1] Noyau fractionnaire")
    t = np.linspace(1e-3, 2.0, 400)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for H in [0.05, 0.1, 0.25, 0.5, 0.75]:
        alpha = H + 0.5
        ax[0].plot(t, fractional_kernel(t, alpha), label=f"$H={H}$")
    ax[0].set(xlabel="$t$", ylabel=r"$K_{1,\alpha}(t) = t^{\alpha-1}/\Gamma(\alpha)$",
              title=r"Noyau fractionnaire, $\alpha = H + 1/2$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend()

    # comparaison avec exponentiel et gamma
    from src import exponential_kernel, gamma_kernel
    ax[1].plot(t, exponential_kernel(t, c=1.0, lam=2.0), label=r"Exp $\lambda=2$")
    ax[1].plot(t, gamma_kernel(t, c=1.0, alpha=0.6, lam=1.0), label=r"Gamma $\alpha=0.6,\lambda=1$")
    ax[1].plot(t, fractional_kernel(t, alpha=0.6), "--", label=r"Fractional $\alpha=0.6$")
    ax[1].set(xlabel="$t$", ylabel=r"$K(t)$", title="Comparaison des noyaux")
    ax[1].legend()
    fig.tight_layout()
    save(fig, "fig01_noyaux")


# ============================================================================
# FIGURE 2 -- Trajectoires du couple (eta, Z, V) comme dans [BCGP23] section 5.3.3
# ============================================================================
def fig_trajectories():
    print("[fig 2] Trajectoires elephant/goldfish")
    T, N = 5.0, 8000
    cases = [
        ("A  $\\eta_0=0$, $\\mu=2,\\nu=1.2,\\theta=0.1$", dict(eta0=0.0, mu=2.0, nu=1.2, theta=0.1)),
        ("B  $\\eta_0=\\mu/\\nu$", dict(eta0=2.0/1.2, mu=2.0, nu=1.2, theta=0.1)),
        ("C  rev. rapide $\\nu=20,\\theta=0.01$", dict(eta0=2.0/20.0, mu=2.0, nu=20.0, theta=0.01)),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for row, (title, kw) in enumerate(cases):
        params = QuadRoughHestonParams(H=0.1, a=0.384, b=0.095, c=0.0025, **kw)
        sim = simulate_elephant_goldfish(params, T, N, n_paths=1,
                                         rng=np.random.default_rng(row + 7),
                                         scheme="stepwise_kernel")
        t = sim["t"]
        axes[row, 0].plot(t, sim["eta"][0], color="tab:blue", lw=0.8)
        axes[row, 0].set_ylabel("$\\eta_t$  (goldfish)")
        axes[row, 0].set_title(title + " -- goldfish")
        axes[row, 1].plot(t, sim["Z"][0], color="tab:red", lw=0.8)
        axes[row, 1].set_title("elephant $Z_t$")
        axes[row, 2].plot(t, sim["V"][0], color="tab:green", lw=0.8)
        axes[row, 2].set_title("variance $V_t = a(\\eta_t-b)^2 + c$")
    for a in axes[-1]:
        a.set_xlabel("$t$")
    fig.tight_layout()
    save(fig, "fig02_trajectoires")


# ============================================================================
# FIGURE 3 -- Role du parametre H sur la rugosite
# ============================================================================
def fig_role_of_H():
    print("[fig 3] Effet de H sur la rugosite de Z")
    T, N = 1.0, 4000
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=False)
    for i, H in enumerate([0.05, 0.1, 0.3, 0.49]):
        params = QuadRoughHestonParams(H=H, mu=0.04, nu=1.0, theta=0.3, eta0=0.04)
        sim = simulate_elephant_goldfish(params, T, N, n_paths=3,
                                         rng=np.random.default_rng(i))
        t = sim["t"]
        for j in range(3):
            axes[i].plot(t, sim["Z"][j], lw=0.7)
        axes[i].set(title=f"$H={H}$", xlabel="$t$")
        if i == 0:
            axes[i].set_ylabel("$Z_t$")
    fig.suptitle("Plus $H$ est petit, plus les trajectoires de $Z$ sont rugueuses")
    fig.tight_layout()
    save(fig, "fig03_role_H")


# ============================================================================
# FIGURE 4 -- Vitesse de convergence forte (independance en H)
# ============================================================================
def fig_strong_convergence():
    print("[fig 4] Convergence forte (peut prendre 1-3 min)")
    T = 1.0
    N_ref = 8192
    n_list = [32, 64, 128, 256, 512, 1024, 2048]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for H in [0.1, 0.25, 0.4]:
        params = QuadRoughHestonParams(H=H, mu=0.04, nu=1.5, theta=0.3, eta0=0.04)
        t0 = time.time()
        res = strong_error_study(params, T, n_list, N_ref, n_paths=800,
                                 p=2.0, rng=np.random.default_rng(int(1000 * H)),
                                 scheme_Z="semi_integrated")
        print(f"  H={H} done in {time.time()-t0:.1f}s")
        axes[0].loglog(res["h_list"], res["err_eta"], "o-", label=f"$H={H}$")
        axes[1].loglog(res["h_list"], res["err_Z"], "o-", label=f"$H={H}$")

    h_ref = np.array([min(res["h_list"]), max(res["h_list"])])
    axes[0].loglog(h_ref, 0.02 * h_ref ** 0.5, "k--", lw=1, label="pente $1/2$")
    axes[1].loglog(h_ref, 0.2 * h_ref ** 0.5, "k--", lw=1, label="pente $1/2$")

    axes[0].set(title="Erreur forte sur $\\eta$ (goldfish)",
                xlabel="$h = T/n$",
                ylabel="$\\|\\sup_t|\\eta^h_t - \\eta^{h_{\\mathrm{ref}}}_t|\\|_{L^2}$")
    axes[1].set(title="Erreur forte sur $Z$ (elephant)",
                xlabel="$h$",
                ylabel="$\\|\\sup_t|Z^h_t - Z^{h_{\\mathrm{ref}}}_t|\\|_{L^2}$")
    axes[0].legend(); axes[1].legend()
    fig.suptitle("Vitesse de convergence forte $\\simeq h^{1/2}$ -- independante du parametre de Hurst $H$")
    fig.tight_layout()
    save(fig, "fig04_convergence_forte")


# ============================================================================
# FIGURE 5 -- Trajectoires de S, V et effet feedback quadratique
# ============================================================================
def fig_asset_and_feedback():
    print("[fig 5] S, V, effet feedback")
    params = QuadRoughHestonParams(H=0.1, mu=0.04, nu=1.0, theta=0.3,
                                   a=0.384, b=0.095, c=0.0025, eta0=0.04)
    T = 1.0
    N = 4000
    sim = simulate_asset(params, T, N, n_paths=5, S0=1.0, rho=-0.7,
                         rng=np.random.default_rng(11),
                         scheme="stepwise_kernel")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i in range(5):
        axes[0].plot(sim["t"], sim["S"][i], lw=0.8, alpha=0.85)
        axes[1].plot(sim["t"], np.sqrt(sim["V"][i]), lw=0.8, alpha=0.85)
    axes[0].set(ylabel="$S_t$", title="Actif sous-jacent : effets de rugosite et feedback negatif")
    axes[1].set(xlabel="$t$", ylabel="$\\sqrt{V_t}$ (volatilite spot)",
                title="Volatilite spot -- pics a la suite de baisses de $S$")
    fig.tight_layout()
    save(fig, "fig05_S_et_V")


# ============================================================================
# FIGURE 6 -- Smile de volatilite implicite
# ============================================================================
def fig_smile():
    print("[fig 6] Smile de vol implicite (~ 30 s / maturite)")
    params = QuadRoughHestonParams(H=0.1, mu=0.04, nu=1.0, theta=0.3,
                                   a=0.384, b=0.095, c=0.0025, eta0=0.04)
    S0 = 1.0
    strikes = np.array([0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.15])
    n_paths = 30000

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for T in [1/12, 3/12, 6/12, 1.0]:
        N = max(200, int(400 * T))
        sim = simulate_asset(params, T, N, n_paths=n_paths, S0=S0, rho=-0.7,
                             rng=np.random.default_rng(int(1000 * T)),
                             scheme="stepwise_kernel")
        smile = smile_from_paths(sim["S"], strikes, T, S0=S0, r=0.0)
        ax.plot(strikes / S0, 100 * smile["iv"], "o-", label=f"$T={T:.3f}$")
    ax.set(xlabel="Moneyness $K/S_0$", ylabel="Vol implicite (%)",
           title="Smile quadratic rough Heston -- asymetrie negative et explosion ATM")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig06_smile")


# ============================================================================
# FIGURE 7 -- Term-structure de l'ATM skew (loi de puissance ?)
# ============================================================================
def fig_atm_skew():
    print("[fig 7] Term-structure du skew ATM (peut prendre plusieurs minutes)")
    params = QuadRoughHestonParams(H=0.1, mu=0.04, nu=1.0, theta=0.3,
                                   a=0.384, b=0.095, c=0.0025, eta0=0.04)
    S0 = 1.0
    strikes_rel = np.array([0.97, 1.0, 1.03])
    Ts = np.array([1/52, 2/52, 1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
    n_paths = 40000

    skew = np.zeros_like(Ts)
    for i, T in enumerate(Ts):
        N = max(200, int(400 * T))
        sim = simulate_asset(params, T, N, n_paths=n_paths, S0=S0, rho=-0.7,
                             rng=np.random.default_rng(100 + i),
                             scheme="stepwise_kernel")
        sm = smile_from_paths(sim["S"], strikes_rel, T, S0=S0, r=0.0)
        skew[i] = (sm["iv"][0] - sm["iv"][2]) / (np.log(strikes_rel[2]) - np.log(strikes_rel[0]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(Ts, skew, "o-")
    axes[0].set(xlabel="Maturite $T$", ylabel="ATM skew",
                title="Structure par terme du skew ATM")
    # Ajustement en loi de puissance T^{H-1/2}
    mask = (Ts > 1/12) & (Ts < 1.5)
    p = np.polyfit(np.log(Ts[mask]), np.log(np.abs(skew[mask])), 1)
    axes[1].loglog(Ts, np.abs(skew), "o", label="donnees simulees")
    axes[1].loglog(Ts, np.exp(p[1]) * Ts ** p[0], "r--",
                   label=f"fit: pente $\\approx{p[0]:.2f}$ (theorie $H-1/2={params.H-0.5:.2f}$)")
    axes[1].set(xlabel="$T$", ylabel="|ATM skew|",
                title="Echelle log-log : verification de la loi de puissance")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "fig07_atm_skew")


# ============================================================================
# FIGURE 8 -- Comparaison des deux schemas d'Euler
# ============================================================================
def fig_scheme_comparison():
    print("[fig 8] Comparaison schemas semi-integrated vs stepwise_kernel")
    params = QuadRoughHestonParams(H=0.1, mu=0.04, nu=1.0, theta=0.3,
                                   a=0.384, b=0.095, c=0.0025, eta0=0.04)
    T = 1.0
    N_ref = 8192
    n_list = [32, 64, 128, 256, 512, 1024, 2048]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for scheme in ["stepwise_kernel", "semi_integrated"]:
        res = strong_error_study(params, T, n_list, N_ref, n_paths=500,
                                 rng=np.random.default_rng(7),
                                 scheme_Z=scheme)
        ax.loglog(res["h_list"], res["err_Z"], "o-", label=scheme)
    h_ref = np.array([T / max(n_list), T / min(n_list)])
    ax.loglog(h_ref, 0.2 * h_ref ** 0.5, "k--", lw=1, label="pente $1/2$")
    ax.set(xlabel="$h$", ylabel="erreur forte sur $Z$",
           title="Comparaison des deux variantes du pseudo-Euler")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig08_scheme_comparison")


if __name__ == "__main__":
    fig_kernel_shapes()
    fig_trajectories()
    fig_role_of_H()
    fig_asset_and_feedback()
    fig_smile()
    fig_strong_convergence()
    fig_atm_skew()
    fig_scheme_comparison()
    print("DONE - toutes les figures generees dans report/figures/")
