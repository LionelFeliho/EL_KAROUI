# Volatilité rugueuse avec trajectoires dépendantes

**Projet de fin d'études – DUFE 2025-2026, sujet 5 (encadré par G. Pagès).**

Implémentation du cadre Bonesini-Callegaro-Grasselli-Pagès (BCGP23) pour la
simulation d'équations de Volterra stochastiques path-dépendantes, appliqué au
modèle **quadratic rough Heston** de Gatheral-Jusselin-Rosenbaum (GJR20).

## Structure du projet

```
PROJET_ROUGH_VOL/
├── src/                              # Paquet Python
│   ├── __init__.py
│   ├── kernels.py                    # noyau fractionnaire K_alpha, co-noyau, burst
│   ├── simulation.py                 # schémas d'Euler hybride (eta, Z, V, S)
│   ├── pricing.py                    # Black-Scholes, IV, smile MC
│   └── convergence.py                # étude de la vitesse de convergence forte
├── notebook/
│   └── rough_volatility_project.ipynb  # Notebook principal (explications + code)
├── report/
│   ├── main.tex                      # Rapport LaTeX (15+ pages)
│   └── figures/                      # Figures générées (PDF + PNG)
├── make_figures.py                   # Script régénérant toutes les figures
└── README.md
```

## Comment exécuter

### 1. Dépendances

Python ≥ 3.10, avec :
- `numpy`
- `scipy`
- `matplotlib`
- `jupyter` (pour le notebook)

```bash
pip install numpy scipy matplotlib jupyter
```

### 2. Générer les figures

Depuis la racine du projet :

```bash
python make_figures.py
```

Cela crée tous les graphiques dans `report/figures/` (≈ 3 minutes).

### 3. Lancer le notebook

```bash
cd notebook
jupyter notebook rough_volatility_project.ipynb
```

### 4. Compiler le rapport LaTeX

Avec **TeXLive** ou **MiKTeX** (nécessaire : `babel-french`, `mathtools`,
`hyperref`, `booktabs`, `microtype`, `listings`, `xcolor`, `subcaption`) :

```bash
cd report
pdflatex main.tex
pdflatex main.tex    # 2e passage pour la table des matières
```

Ou via **Overleaf** : téléverser le dossier `report/` et compiler en pdfLaTeX.

## Aperçu du contenu

### Théorie
1. Motivation de la volatilité rugueuse (estimation empirique $H \approx 0{,}1$).
2. Mouvement Brownien fractionnaire, représentation de Riemann-Liouville.
3. Équations de Volterra stochastiques path-dépendantes.
4. Théorème de pont BCGP : éléphant ↔ poisson-rouge.
5. Schémas d'Euler hybrides (stepwise kernel, semi-integrated) et vitesse $h^{1/2}$
   indépendante de $H$.

### Expériences numériques
- Trajectoires du couple $(\eta, Z, V)$ : trois régimes selon $\eta_0$.
- Effet du paramètre de Hurst $H$ sur la rugosité.
- **Vérification numérique de la convergence forte $h^{1/2}$** pour $H \in \{0{,}1, 0{,}25, 0{,}4\}$.
- Comparaison des deux variantes du schéma.
- Trajectoires de l'actif $S$ et de la variance $V$ : effet feedback Zumbach.
- Smile de volatilité implicite (Monte-Carlo sur 30 000 trajectoires).
- Structure par terme du skew ATM : loi de puissance $T^{H-1/2}$.

### Propositions de recherche (détaillées en §7 du rapport)
1. Approximation multi-facteurs du noyau (Abi Jaber-El Euch).
2. Quantification fonctionnelle pour le pricing d'exotiques.
3. Calibration jointe SPX/VIX vs. Quintic OU.
4. Greeks par Malliavin via le poisson-rouge.
5. Extensions path-dépendantes complètes.
6. Réduction de variance avec variable de contrôle Black-Scholes.

## Références

- **[BCGP23]** Bonesini, Callegaro, Grasselli, Pagès.
  *From elephant to goldfish (and back): memory in stochastic Volterra processes.*
  arXiv:2306.02708, 2023.
- **[GJR20]** Gatheral, Jusselin, Rosenbaum.
  *The quadratic rough Heston model and the joint S&P500/VIX smile calibration.*
  arXiv:2001.01789, 2020.
- **[GJR14]** Gatheral, Jaisson, Rosenbaum. *Volatility is rough.* QF, 2018.

## Auteur

Lionel Feliho – DUFE 2025-2026 – Sorbonne Université.
