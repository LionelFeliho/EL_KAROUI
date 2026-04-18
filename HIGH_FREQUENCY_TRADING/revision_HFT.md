# Révisions HFT — Préparation à l'examen rédactionnel

> **Cours de Sophie Laruelle (LAMA-UPEC)** — *High Frequency Data in Finance*
> Document structuré pour ~10–12 h de lecture. Ordre inversé demandé : **HFT4 → HFT3 → HFT2 → HFT1**.
> Chaque concept est expliqué avec : (i) sa définition mathématique, (ii) l'intuition, (iii) pourquoi on utilise cette formule, (iv) ce qu'il faut savoir expliquer à l'examen.

---

## Conseils méthodologiques pour un examen rédactionnel

1. **N'écris jamais une formule sans dire ce qu'elle modélise.** Le correcteur cherche surtout si tu as compris le *pourquoi*.
2. **Structure chaque réponse** : définition → hypothèses → formule → interprétation économique/financière → limitations.
3. **Pour les formules complexes** (Hawkes 2D, covariance de HY, etc.) : mieux vaut écrire la version 1D correctement et dire « l'extension multidimensionnelle suit le même principe mais avec une matrice de noyaux Φ » que mal recopier de mémoire une formule matricielle.
4. Les quatre questions **obligatoires** à maîtriser d'après la demande :
   - Comment volume / spread / volatilité évoluent au cours de la journée (courbes intraday) ;
   - Les estimateurs de volatilité et corrélation en haute fréquence (et pourquoi les « naïfs » échouent) ;
   - Le processus de Hawkes en dimension 1 (définition, stationnarité, simulation, MLE) ;
   - Le processus de Hawkes en dimension 2 (rayon spectral, cross-excitation, modèle de prix de Bacry).

---

# PARTIE I — HFT4 : Processus de Hawkes et Microstructure

Le cours HFT4 est le plus technique. Il propose une modélisation unifiée des événements de marché (trades, limit orders, cancellations) par des processus ponctuels **auto-excités**. L'idée fondamentale : contrairement à un Poisson où l'intensité est constante (ou déterministe), dans un Hawkes, **chaque événement passé augmente la probabilité d'un nouvel événement** — ce qui produit des *clusters* (amas temporels), exactement ce qu'on observe empiriquement en microstructure.

## Chapitre 1 — Introduction aux processus ponctuels

### 1.1 Processus ponctuel simple

**Définition.** Sur un espace probabilisé $(\Omega, \mathcal{A}, \mathbb{P})$, une suite de v.a. positives $(\tau_i)_{i \geq 1}$ strictement croissante ($\tau_i < \tau_{i+1}$) est appelée *processus ponctuel simple* sur $\mathbb{R}_+$.

**Intuition.** Les $\tau_i$ représentent les instants d'occurrence d'événements : arrivées de trades, d'ordres limite, de cancellations, etc. « Simple » veut dire qu'il n'y a pas deux événements au même instant (pas de saut de taille ≥ 2).

### 1.2 Processus de comptage et durée

**Processus de comptage** associé à $(\tau_i)$ :
$$N_t = \sum_{i \geq 1} \mathbf{1}_{\tau_i \leq t}$$
C'est une fonction en escalier, continue à droite, qui fait un saut de +1 à chaque $\tau_i$. $N_t$ compte le nombre d'événements survenus sur $[0, t]$.

**Processus de durée** :
$$\Delta \tau_i = \tau_i - \tau_{i-1}$$
Ce sont les temps entre événements (inter-arrival times). À l'examen, sache bien expliquer que **la loi des $\Delta \tau_i$ caractérise le processus** : i.i.d. exponentielles ⇒ Poisson ; distributions variées ⇒ autre chose.

### 1.3 Processus d'intensité

**Définition.** Soit $N$ un processus de comptage adapté à $(\mathcal{F}_t)$. L'intensité (continue à gauche) est :
$$\lambda(t | \mathcal{F}_t) = \lim_{h \downarrow 0} \mathbb{E}\!\left[\frac{N_{t+h} - N_t}{h} \,\Big|\, \mathcal{F}_t\right] = \lim_{h \downarrow 0} \frac{1}{h} \mathbb{P}(N_{t+h} - N_t > 0 \mid \mathcal{F}_t)$$

**Pourquoi cette double formulation ?** La première dit : « combien d'événements en moyenne par unité de temps conditionnellement au passé ». La seconde (équivalente pour un processus simple) dit : « probabilité qu'au moins un événement arrive dans le petit intervalle suivant ». Elles coïncident parce que pour un processus simple, $\mathbb{E}[N_{t+h}-N_t] \approx \mathbb{P}(N_{t+h}-N_t \geq 1)$ quand $h \to 0$.

On note simplement $\lambda(t)$ lorsque la filtration est la filtration naturelle $\mathcal{F}_t^N$ de $N$.

### 1.4 Poisson homogène — l'exemple de référence

Pour $\lambda \in \mathbb{R}_+^*$ constant :
$$\mathbb{P}(N_{t+h} - N_t = 1 | \mathcal{F}_t) = \lambda h + o(h), \qquad \mathbb{P}(N_{t+h} - N_t > 1 | \mathcal{F}_t) = o(h)$$

**Trois propriétés-clés à citer à l'examen :**
1. L'intensité **ne dépend pas** du passé → **pas de mémoire**, pas de clustering.
2. Les durées $\Delta \tau_i$ sont i.i.d. de loi exponentielle $\mathcal{E}(\lambda)$.
3. Processus **markovien** → très utilisé en théorie des files d'attente (M/M/1, etc.).

**Pourquoi insuffisant en finance ?** Parce qu'empiriquement, les trades arrivent en grappes (clusters) : après un trade, la probabilité d'en observer un autre rapidement augmente. Un Poisson homogène prédit l'inverse. D'où la nécessité de processus auto-excités.

### 1.5 Changement de temps stochastique (théorème-clé)

**Intensité intégrée** :
$$\Lambda(\tau_{i-1}, \tau_i) = \int_{\tau_{i-1}}^{\tau_i} \lambda(s)\, ds$$

**Théorème de changement de temps.** Tout processus de comptage est un processus de Poisson d'intensité 1 après un changement de temps approprié. Formellement : si $N$ a une intensité $\lambda$, alors $\widetilde N_\tau = N_{t_\tau}$ avec $\int_0^{t_\tau} \lambda(s) ds = \tau$ est un Poisson standard.

**Conséquence pratique (à l'examen !).** Ce théorème est l'outil universel pour **tester** si on a bien simulé ou bien estimé un Hawkes : on calcule les $\Lambda(\tau_{i-1}, \tau_i)$ et on vérifie qu'ils sont i.i.d. $\mathcal{E}(1)$ via un QQ-plot.

> **Analogie à retenir :** c'est l'équivalent pour les processus de comptage du théorème de Dubins-Schwarz pour les martingales (toute martingale continue est un mouvement brownien changé de temps).

---

## Chapitre 2 — Processus de Hawkes unidimensionnel

### 2.1 Définition — processus auto-excitant linéaire

**Définition générale.** Un processus $N$ est un Hawkes linéaire si son intensité vérifie :
$$\boxed{\lambda(t) = \lambda_0(t) + \int_{-\infty}^t \varphi(t-s)\, dN_s = \lambda_0(t) + \sum_{\tau_i < t} \varphi(t - \tau_i)}$$

où :
- $\lambda_0 : \mathbb{R} \to \mathbb{R}_+$ est l'**intensité de base** (« baseline »), déterministe ;
- $\varphi : \mathbb{R}_+ \to \mathbb{R}_+$ est le **noyau** (« kernel ») qui mesure l'influence positive des événements passés sur l'intensité présente.

**Intuition à maîtriser parfaitement.** À chaque événement passé $\tau_i$, on ajoute une contribution $\varphi(t - \tau_i)$ à l'intensité courante. Plus $\tau_i$ est récent, plus $\varphi(t-\tau_i)$ est grand (si $\varphi$ est décroissante). Donc un événement récent « excite » les arrivées futures → clustering.

### 2.2 Cas standard : noyau exponentiel (Hawkes 1971)

Hawkes propose le noyau exponentiel :
$$\varphi(t) = \sum_{j=1}^P \alpha_j e^{-\beta_j t} \mathbf{1}_{\mathbb{R}_+}(t)$$

D'où :
$$\lambda(t) = \lambda_0(t) + \sum_{j=1}^P \sum_{\tau_i < t} \alpha_j e^{-\beta_j (t - \tau_i)}$$

**Version simplifiée** (P = 1, $\lambda_0$ constant), celle qu'on utilise 90 % du temps :
$$\boxed{\lambda(t) = \lambda_0 + \alpha \sum_{\tau_i < t} e^{-\beta(t - \tau_i)}}$$

**Pourquoi exponentiel ?**
- **Paramètres interprétables** : $\alpha$ = intensité du saut de l'excitation à chaque événement (combien ça « augmente » l'intensité d'un coup) ; $\beta$ = vitesse de décroissance de la mémoire (grand $\beta$ ⇒ oubli rapide).
- **Markovianité** : le couple $(N_t, \lambda(t))$ devient markovien, ce qui permet calculs explicites et simulation efficace.
- Permet l'écriture **récursive** des log-vraisemblances et des intensités intégrées (cf. sections suivantes).

**Limite à mentionner en culture générale** : les études empiriques récentes (Bacry et al. 2012, 2016) montrent que le noyau réel des marchés est mieux modélisé par une **loi de puissance** que par une exponentielle. L'exponentielle reste néanmoins le standard pédagogique et pour l'estimation.

### 2.3 Stationnarité — condition fondamentale

**Cas général.** Supposons $\mathbb{E}[\lambda(t)] = \mu$ constant. Alors :
$$\mu = \lambda_0 + \mathbb{E}\!\left[\int_{-\infty}^t \varphi(t-s)\, dN_s\right] = \lambda_0 + \int_{-\infty}^t \varphi(t-s)\, \mathbb{E}[\lambda(s)]\, ds = \lambda_0 + \mu \int_0^\infty \varphi(u)\, du$$

D'où, si $\int_0^\infty \varphi(u) du < 1$ :
$$\boxed{\mu = \frac{\lambda_0}{1 - \int_0^\infty \varphi(u)\, du}}$$

**Détail du calcul à savoir refaire.** Deux étapes astucieuses :
1. On utilise que $\mathbb{E}[dN_s] = \mathbb{E}[\lambda(s)] ds$ (propriété du compensateur) pour remplacer $dN_s$ par $\lambda(s) ds$ sous l'espérance.
2. On change de variable $u = t-s$ dans l'intégrale, et on utilise la stationnarité ($\mathbb{E}[\lambda(s)] = \mu$ pour tout $s$).

**Cas exponentiel.** $\int_0^\infty \alpha_j e^{-\beta_j u} du = \alpha_j / \beta_j$, donc la condition de stationnarité s'écrit :
$$\boxed{\sum_{j=1}^P \frac{\alpha_j}{\beta_j} < 1}$$

Et, pour $P=1$, l'intensité moyenne stationnaire vaut :
$$\boxed{\mathbb{E}[\lambda(t)] = \frac{\lambda_0}{1 - \alpha/\beta}}$$

**Interprétation économique essentielle.** Le rapport $\alpha/\beta$ est appelé **« branching ratio »** (rapport de branchement). Il représente le nombre moyen d'événements qu'un événement donné va engendrer directement (par auto-excitation). Si ce rapport est ≥ 1, l'excitation s'auto-entretient à l'infini et le système « explose » (non-stationnaire). Si < 1, l'excitation s'atténue avec le temps. Empiriquement sur les marchés, ce rapport est **proche de 1** (parfois 0.9+), ce qui signale un marché **fortement endogène**.

### 2.4 Simulation — algorithme d'Ogata (1981)

**Thinning procedure (Lewis & Shedler 1979).** Pour simuler un processus ponctuel d'intensité $\lambda(t)$ bornée par $\lambda^*(t)$ :
1. Simuler un Poisson inhomogène de taux $\lambda^*(t)$.
2. Pour chaque point $\tau_i^*$ simulé, le **conserver avec probabilité $\lambda(\tau_i^*)/\lambda^*(\tau_i^*)$** (équivalent : l'**éliminer** avec probabilité $1 - \lambda(\tau_i^*)/\lambda^*(\tau_i^*)$).

Le résultat est un processus ponctuel inhomogène d'intensité $\lambda(t)$. C'est le principe d'**acceptation-rejet** appliqué aux processus ponctuels.

**Algorithme d'Ogata spécialisé pour Hawkes** (pour $P=1$, $[0,T]$) :

1. **Initialisation** : $\lambda^* \leftarrow \lambda_0$, $n \leftarrow 1$.
2. **Premier événement** : tirer $U \sim \mathcal{U}_{[0,1]}$, poser $s \leftarrow -\frac{1}{\lambda^*}\ln U$. Si $s \leq T$, $\tau_1 \leftarrow s$, sinon sortir.
3. **Routine générale** : $n \leftarrow n+1$.
   a. **Mise à jour de l'intensité majorante** : $\lambda^* \leftarrow \lambda(\tau_{n-1}) + \alpha$ (le « + α » représente le saut instantané de l'intensité dû à l'événement qui vient juste d'arriver — comme $\lambda$ est continue à gauche, ce saut ne figure pas dans $\lambda(\tau_{n-1})$).
   b. **Nouvel événement candidat** : tirer $U \sim \mathcal{U}_{[0,1]}$, poser $s \leftarrow s - \frac{1}{\lambda^*}\ln U$. Si $s > T$, sortir.
   c. **Test de rejet** : tirer $D \sim \mathcal{U}_{[0,1]}$.
      - Si $D \leq \lambda(s)/\lambda^*$ : accepter, $\tau_n \leftarrow s$, retour à 3.
      - Sinon : $\lambda^* \leftarrow \lambda(s)$ (on rétrécit la majorante car, sans nouvel événement, $\lambda$ a décru), retour à 3b.

**Pourquoi $s \leftarrow s - \frac{1}{\lambda^*}\ln U$ ?** Parce que si $U \sim \mathcal{U}_{[0,1]}$, alors $-\ln U / \lambda^* \sim \mathcal{E}(\lambda^*)$ : c'est la durée jusqu'au prochain événement d'un Poisson homogène de taux $\lambda^*$. Tant qu'on n'a pas d'événement, $\lambda$ décroît, mais on peut majorer $\lambda$ par la valeur au dernier « checkpoint » $\lambda^*$ et faire du thinning.

### 2.5 Test du processus simulé — propriété de changement de temps

Par le théorème de changement de temps, les $\Lambda(\tau_{i-1}, \tau_i) = \int_{\tau_{i-1}}^{\tau_i}\lambda(s) ds$ doivent être i.i.d. $\mathcal{E}(1)$.

**Calcul explicite pour Hawkes exponentiel** (à savoir refaire !). En partant de $\int_{\tau_{i-1}}^{\tau_i} \lambda(s)\,ds$ et en séparant la baseline et la partie auto-excitante :
$$\Lambda(\tau_{i-1}, \tau_i) = \int_{\tau_{i-1}}^{\tau_i} \lambda_0(s)\, ds + \sum_{\tau_k \leq \tau_{i-1}} \frac{\alpha}{\beta}\!\left[e^{-\beta(\tau_{i-1} - \tau_k)} - e^{-\beta(\tau_i - \tau_k)}\right]$$

**Forme récursive élégante.** Posons $A_i = \sum_{\tau_k \leq \tau_i} e^{-\beta(\tau_i - \tau_k)}$. Alors :
$$A_i = 1 + e^{-\beta(\tau_i - \tau_{i-1})} A_{i-1}, \qquad A_0 = 0$$

(Le « +1 » correspond au dernier événement $\tau_i$ lui-même ; le reste décroît du facteur $e^{-\beta(\tau_i - \tau_{i-1})}$.) D'où :
$$\boxed{\Lambda(\tau_{i-1}, \tau_i) = \int_{\tau_{i-1}}^{\tau_i} \lambda_0(s)\, ds + \frac{\alpha}{\beta}\!\left[1 - e^{-\beta(\tau_i - \tau_{i-1})}\right] A_{i-1}}$$

**Test pratique (à mentionner à l'examen).** On trace le **QQ-plot** des $\Lambda(\tau_{i-1}, \tau_i)$ contre les quantiles d'une loi $\mathcal{E}(1)$. Si le modèle est bon, on obtient une droite de pente 1. Sinon, c'est que le modèle ne capture pas correctement la dynamique des données.

### 2.6 Estimation par maximum de vraisemblance

**Log-vraisemblance générale pour un processus ponctuel d'intensité $\lambda$** observé sur $[0, T]$ :
$$\boxed{\ln L((N_t)_{t \in [0,T]}) = \int_0^T (1 - \lambda(s))\, ds + \int_0^T \ln \lambda(s)\, dN_s}$$

**Comment obtenir cette formule ?** Densité d'un processus ponctuel par rapport à la mesure d'un Poisson standard. Le premier terme $\int(1-\lambda) ds$ pénalise les intervalles où l'intensité est forte mais sans événement (c'est le « compensateur »). Le second terme $\int \ln \lambda\, dN_s = \sum_i \ln \lambda(\tau_i)$ récompense les événements survenus là où l'intensité était élevée.

**Spécialisation Hawkes** ($\tau_n$ dernier événement observé) :
$$\ln L((\tau_i)_{i=1,\ldots,n}) = \tau_n - \Lambda(0, \tau_n) + \sum_{i=1}^n \ln \lambda(\tau_i) = \tau_n - \Lambda(0, \tau_n) + \sum_{i=1}^n \ln\!\left(\lambda_0(\tau_i) + \alpha \sum_{k=1}^{i-1} e^{-\beta(\tau_i - \tau_k)}\right)$$

**Forme récursive** (Ogata). Posons $R_i = \sum_{k=1}^{i-1} e^{-\beta(\tau_i - \tau_k)}$. Alors :
$$R_i = e^{-\beta(\tau_i - \tau_{i-1})}(1 + R_{i-1}), \quad R_0 = 0$$

D'où finalement (avec $\lambda_0$ constant) :
$$\boxed{\ln L = \tau_n - \lambda_0 \tau_n - \sum_{i=1}^n \frac{\alpha}{\beta}\!\left[1 - e^{-\beta(\tau_n - \tau_i)}\right] + \sum_{i=1}^n \ln(\lambda_0 + \alpha R_i)}$$

**Pourquoi la récursion ?** Sans elle, le calcul naïf de $\sum_{i=1}^n \sum_{k=1}^{i-1} \ldots$ est en $O(n^2)$. Avec la récursion, c'est $O(n)$. C'est essentiel car les jeux de données de microstructure ont typiquement $n = 10^5$ à $10^7$ événements.

**Propriétés asymptotiques de l'estimateur MLE** (Ogata 1978). Pour un Hawkes 1D stationnaire avec $\lambda_0$ constant et $P=1$, $\hat\theta_T = (\hat\lambda_0, \hat\alpha, \hat\beta)$ est :
- **Consistant** : convergence en probabilité vers $\theta$ quand $T \to \infty$.
- **Asymptotiquement normal** : $\sqrt{T}(\hat\theta - \theta) \xrightarrow{\mathcal{L}} \mathcal{N}(0, I^{-1}(\theta))$ où $I^{-1}(\theta) = \mathbb{E}\!\left[\frac{1}{\lambda}\frac{\partial \lambda}{\partial \theta_i}\frac{\partial \lambda}{\partial \theta_j}\right]_{i,j}$ est la matrice d'information de Fisher.
- **Asymptotiquement efficace** : atteint la borne de Cramér-Rao.

**Résultats numériques de référence** (Ogata). Avec $\lambda_0 = 1.2, \alpha = 0.6, \beta = 0.8$ et T variable :
- T = 100 : écart-types ~ 0.37, 0.16, 0.44 (trop peu de données)
- T = 10000 : écart-types ~ 0.045, 0.016, 0.023 (bonne précision)
- T = 100000 : écart-types ~ 0.014, 0.004, 0.007 (quasi-exact)

L'algorithme utilisé est typiquement Nelder-Mead (optimisation sans gradient, robuste).

---

## Chapitre 3 — Processus de Hawkes multidimensionnels

### 3.1 Définition

Soit $M \in \mathbb{N}^*$ la dimension. On considère $M$ processus ponctuels $\{(\tau_i^m)_i\}_{m=1,\ldots,M}$ et leurs processus de comptage $\mathbf{N}_t = (N_t^1, \ldots, N_t^M)$.

**Un Hawkes multidimensionnel** a pour intensités (pour $m = 1, \ldots, M$) :
$$\boxed{\lambda^m(t) = \lambda_0^m(t) + \sum_{n=1}^M \int_0^t \sum_{j=1}^P \alpha_j^{mn} e^{-\beta_j^{mn}(t-s)}\, dN_s^n}$$

Version simplifiée ($P = 1$, $\lambda_0^m$ constant) :
$$\lambda^m(t) = \lambda_0^m + \sum_{n=1}^M \sum_{\tau_i^n < t} \alpha^{mn} e^{-\beta^{mn}(t - \tau_i^n)}$$

**Interprétation des indices $mn$ :** $\alpha^{mn}$ mesure combien la composante $m$ est excitée par un événement de la composante $n$.
- $m = n$ : **auto-excitation** (un événement de type $m$ augmente sa propre intensité).
- $m \neq n$ : **excitation croisée** (cross-excitation) — événements d'un type influent sur un autre.

### 3.2 Condition de stationnarité

En notation matricielle :
$$\boldsymbol\lambda(t) = \boldsymbol\lambda_0 + \int_0^t \boldsymbol\Phi(t-s)\, d\mathbf{N}_s, \qquad \Phi(t) = \left(\alpha^{mn} e^{-\beta^{mn} t}\right)_{m,n}$$

En supposant stationnarité, $\mathbb{E}[\boldsymbol\lambda(t)] = \boldsymbol\mu$ vecteur constant, et :
$$\boxed{\boldsymbol\mu = \left(\mathbf{I} - \int_0^\infty \Phi(u)\, du\right)^{-1} \boldsymbol\lambda_0}$$

**Condition suffisante de stationnarité** : le **rayon spectral** de la matrice
$$\boldsymbol\Gamma = \int_0^\infty \Phi(u)\, du = \left(\frac{\alpha^{mn}}{\beta^{mn}}\right)_{m,n}$$
doit être **strictement inférieur à 1** :
$$\boxed{\rho(\boldsymbol\Gamma) = \max\{|a| : a \in \text{Sp}(\boldsymbol\Gamma)\} < 1}$$

**À l'examen**, distingue bien :
- En 1D : $\alpha/\beta < 1$.
- En multi-D : $\rho(\boldsymbol\Gamma) < 1$ (condition sur le rayon spectral, PAS sur chaque entrée de la matrice !).

Cette condition est l'analogue pour les Hawkes de la condition sur les valeurs propres en AR(1) multivarié : elle assure que la dynamique ne s'auto-amplifie pas à l'infini.

### 3.3 Simulation d'un Hawkes multivarié

On généralise Ogata. Soit $I^K(t) = \sum_{n=1}^K \lambda^n(t)$ et $I^M(t)$ l'intensité totale. Algorithme :

1. **Initialisation** : $i \leftarrow 1$, $i^n \leftarrow 1$ pour tout $n$, $I^* \leftarrow I^M(0) = \sum_n \lambda_0^n$.
2. **Premier événement** : $U \sim \mathcal{U}_{[0,1]}$, $s \leftarrow -\frac{1}{I^*}\ln U$.
   - Si $s > T$, sortir.
   - **Attribution** : tirer $D \sim \mathcal{U}_{[0,1]}$ et déterminer $n_0$ tel que $\frac{I^{n_0-1}(0)}{I^*(0)} < D \leq \frac{I^{n_0}(0)}{I^*(0)}$.
   - $\tau_1 \leftarrow \tau_1^{n_0} \leftarrow s$.
3. **Routine générale** : $i^{n_0} \leftarrow i^{n_0}+1$, $i \leftarrow i+1$.
   a. $I^* \leftarrow I^M(t_{i-1}) + \sum_{n=1}^M \sum_{j=1}^P \alpha_j^{nn_0}$ (saut d'intensité dû à l'événement sur la composante $n_0$).
   b. $U \sim \mathcal{U}_{[0,1]}$, $s \leftarrow s - \frac{1}{I^*}\ln U$. Si $s > T$, sortir.
   c. **Attribution-rejet** : $D \sim \mathcal{U}_{[0,1]}$. Si $D \leq I^M(s)/I^*$ : accepter, puis trouver $n_0$ tel que $I^{n_0-1}(s)/I^*(s) < D \leq I^{n_0}(s)/I^*(s)$ et poser $\tau_i^{n_0} \leftarrow s$. Sinon : $I^* \leftarrow I^M(s)$ et retour à 3b.

**Idée-clé** : d'abord on décide *quand* aura lieu le prochain événement (comme en 1D), puis *à quelle composante* il appartient (test d'attribution proportionnel aux intensités).

### 3.4 Log-vraisemblance multivariée et MLE

La log-vraisemblance totale est la somme des log-vraisemblances composante par composante :
$$\ln L(\{\tau_i\}_{i=1,\ldots,N}) = \sum_{m=1}^M \ln L^m(\{\tau_i\})$$
$$\ln L^m(\{\tau_i\}) = \int_0^T (1 - \lambda^m(s))\, ds + \int_0^T \ln \lambda^m(s)\, dN_s^m$$

Forme explicite (avec $z_i^m = \mathbf{1}\{\tau_i \text{ est de type } m\}$) :
$$\ln L^m = T - \Lambda^m(0, T) + \sum_{i=1}^N z_i^m \ln\!\left(\lambda_0^m(\tau_i) + \sum_{n=1}^M \sum_{j=1}^P \alpha_j^{mn} \sum_{\tau_k^n < \tau_i} e^{-\beta_j^{mn}(\tau_i - \tau_k^n)}\right)$$

**Récursivité** pour $R_l^{mn} = \sum_{\tau_k^n < \tau_l^m} e^{-\beta^{mn}(\tau_l^m - \tau_k^n)}$ :
$$R_l^{mn} = \begin{cases} e^{-\beta^{mn}(\tau_l^m - \tau_{l-1}^m)} R_{l-1}^{mn} + \displaystyle\sum_{\tau_{l-1}^m \leq \tau_k^n < \tau_l^m} e^{-\beta^{mn}(\tau_l^m - \tau_k^n)} & \text{si } m \neq n \\ e^{-\beta^{mn}(\tau_l^m - \tau_{l-1}^m)}(1 + R_{l-1}^{mn}) & \text{si } m = n \end{cases}$$

**Attention au cas $m \neq n$** : entre $\tau_{l-1}^m$ et $\tau_l^m$, il peut y avoir des événements de type $n$ (pas seulement à $\tau_l^n = \tau_l^m$), donc on doit ajouter leurs contributions.

**Estimation en pratique** — exemple Hawkes 2D avec 10 paramètres ($\lambda_0^1, \alpha^{11}, \beta^{11}, \alpha^{12}, \beta^{12}$ et idem pour la composante 2) : convergence correcte à partir de T = 500–1000. À T = 100, les écarts-types sont encore très grands (peu d'information dans les données). C'est une limitation pratique : il faut **beaucoup** de données pour estimer un Hawkes multivarié, et plus la dimension est grande, plus il faut de données.

---

## Chapitre 4 — Modélisation du « bruit de microstructure » par Hawkes

C'est la partie *applicative* qui fait la valeur du cours. Le message central : les processus de Hawkes reproduisent **naturellement** plusieurs faits stylisés de microstructure.

### 4.1 Un modèle 1D du prix (Bacry, Delattre, Hoffmann, Muzy 2013)

Le prix est écrit comme :
$$\boxed{p(t) = N^1(t) - N^2(t)}$$

où $N^1$ et $N^2$ sont deux processus de Hawkes 2D avec **uniquement de l'excitation croisée** (pas d'auto-excitation) :
$$\lambda^1(t) = \lambda_0 + \int_{-\infty}^t \alpha e^{-\beta(t-s)} dN_s^2, \qquad \lambda^2(t) = \lambda_0 + \int_{-\infty}^t \alpha e^{-\beta(t-s)} dN_s^1$$

**Trois hypothèses de construction à retenir :**
1. **Pas d'auto-excitation** : une hausse ne déclenche pas d'autre hausse, une baisse ne déclenche pas d'autre baisse.
2. **Uniquement de la cross-excitation** : une hausse déclenche une baisse et réciproquement — cela modélise la **mean-reversion** empirique observée sur les prix haute fréquence.
3. **Symétrie** : $\alpha^{12} = \alpha^{21} = \alpha$, $\beta^{12} = \beta^{21} = \beta$ (pas d'asymétrie haut/bas).

**Pourquoi ce modèle est-il remarquable ?** Parce qu'il produit un **signature plot** explicite.

### 4.2 Volatility Signature Plot — la formule explicite

Le *volatility signature plot* représente la variance réalisée comme fonction du pas d'échantillonnage $\tau$ :
$$RV(\tau) = \frac{1}{\tau} \sum_{i=1}^I (p(i\tau) - p((i-1)\tau))^2$$

Bacry et al. (2011) démontrent que dans le modèle ci-dessus, la variance théorique vaut :
$$\boxed{C(\tau) = \frac{1}{\tau}\mathbb{E}[p(\tau)^2] = \Lambda\!\left[\kappa^2 + (1-\kappa^2)\frac{1 - e^{-\gamma \tau}}{\gamma \tau}\right]}$$

avec :
$$\Lambda = \frac{2 \lambda_0}{1 - \alpha/\beta}, \qquad \kappa = \frac{1}{1 + \alpha/\beta}, \qquad \gamma = \alpha + \beta$$

**Interprétation essentielle** (à comprendre absolument pour l'examen) :

- Quand $\tau \to \infty$ : $(1 - e^{-\gamma \tau})/(\gamma \tau) \to 0$, donc $C(\tau) \to \Lambda \kappa^2$ = variance long terme.
- Quand $\tau \to 0$ : $(1 - e^{-\gamma \tau})/(\gamma \tau) \to 1$, donc $C(\tau) \to \Lambda$ = variance haute fréquence.

Or $\Lambda > \Lambda \kappa^2$ (puisque $\kappa < 1$). Donc **la volatilité mesurée est PLUS grande en haute fréquence qu'en basse fréquence** : c'est exactement le *signature plot* observé empiriquement ! Le modèle Hawkes cross-exciting en donne une explication microstructurelle : la mean-reversion haute fréquence crée une sur-estimation de la volatilité à fine échelle.

### 4.3 Estimation MLE dans ce modèle

Les log-vraisemblances $L_1(\theta), L_2(\theta)$ issues de l'observation respectivement de $N^1$ et $N^2$ :
$$L_1(\theta) = \sum_{0 \leq \tau_i^{(1)} < N^1(T)} \ln\!\left(\lambda_0 + \!\!\sum_{0 \leq \tau_j^{(2)} < N^2(T)}\!\! \alpha e^{-\beta(\tau_i^{(1)} - \tau_j^{(2)})}\right) - (\lambda_0 - 1)T - \sum_{0 \leq \tau_j^{(2)} < N^2(T)} \frac{\alpha}{\beta}(1 - e^{-\beta(T - \tau_j^{(2)})})$$

et symétriquement pour $L_2$. L'estimateur MLE est :
$$\hat\theta_{MLE} = \arg\min_\theta (L_1(\theta) + L_2(\theta))$$

**sous les contraintes** : $\lambda_0 > 0$, $\alpha > 0$, $\beta > 0$, et **stabilité** $\alpha/\beta < 1$. Cette dernière contrainte est cruciale : sans elle, le modèle est non-stationnaire et l'estimation diverge.

### 4.4 Extension 2D — modèle bivarié pour deux actifs

Bacry et al. (2013) proposent :
$$p^1(t) = N^1(t) - N^2(t), \qquad p^2(t) = N^3(t) - N^4(t)$$

où $\mathbf{N} = (N^1, \ldots, N^4)$ est un Hawkes 4D avec intensité gouvernée par une matrice bloc :
$$\lambda(t) = \lambda_0 + \int_0^t \begin{pmatrix} 0 & \varphi_{12} & \varphi_{13} & 0 \\ \varphi_{12} & 0 & 0 & \varphi_{13} \\ \varphi_{31} & 0 & 0 & \varphi_{34} \\ 0 & \varphi_{31} & \varphi_{34} & 0 \end{pmatrix} (t-s) d\mathbf{N}_s$$

**Lecture du design** :
- Pas d'auto-excitation : tous les $\varphi_{ii} = 0$.
- Symétrie haut/bas au sein de chaque actif : $\varphi_{13} = \varphi_{24}$ (une hausse de $p^1$ influe sur les mouvements de $p^2$ de la même manière selon qu'on parle de haut ou de bas), $\varphi_{31} = \varphi_{42}$.
- Les prix s'influencent **positivement** entre eux, pas négativement : $\varphi_{14} = \varphi_{41} = \varphi_{23} = \varphi_{32} = 0$ (une hausse d'un prix ne déclenche pas mécaniquement une baisse de l'autre).

### 4.5 L'effet Epps reproduit par ce modèle 2D

Le **coefficient de corrélation** entre les rendements des deux prix a une forme explicite (Bacry et al. 2011, Prop. 3.1) :
$$\rho(\tau) = \text{Corr}(p^1(t+\tau) - p^1(t), p^2(t+\tau) - p^2(t)) = \frac{C_{12}(\tau)}{C_{11}(\tau)}$$

Les comportements limites à retenir :
$$\rho(\tau) \sim \frac{R(Q_2 - Q_1)}{4\Lambda} \tau + O(\tau^2) \text{ quand } \tau \to 0, \qquad \rho(\tau) \to \frac{2\Gamma_{13}(1 + \Gamma_{12})}{1 + \Gamma_{13}^2 + 2\Gamma_{12} + \Gamma_{12}^2} \text{ quand } \tau \to \infty$$

(avec $\Gamma_{ij} = \alpha^{ij}/\beta^{ij}$.)

**Interprétation fondamentale** : la corrélation en TRÈS haute fréquence ($\tau \to 0$) tend vers 0 (c'est l'**effet Epps** : on ne voit pas de corrélation quand on observe à très courte échelle). Elle augmente avec $\tau$ et se stabilise à une valeur de long terme. Le modèle Hawkes 2D reproduit donc **à la fois** la saturation long-terme et la chute courte-terme de la corrélation empirique.

### 4.6 Extensions — cadre plus riche

**Noyaux en loi de puissance.** Empiriquement (Bacry, Dayri, Muzy 2012), les noyaux réels ressemblent plus à $\varphi(t) \propto t^{-\gamma}$ qu'à des exponentielles. On perd la markovianité, mais on capture mieux la mémoire longue des marchés.

**Limite d'échelle — Jaisson & Rosenbaum (2015).** En renormalisant les Hawkes quand $\sum \alpha_j/\beta_j \to 1^-$ (limite « presque instable »), on obtient, à la limite d'échelle, un processus de Heston à volatilité stochastique :
$$dC_t = \left(\frac{2\mu}{\lambda} - C_t m\right) dt + m_1 \sqrt{C_t}\, dB_t^1, \qquad dP_t = \frac{1}{1 - \|\varphi\|_1}\sqrt{C_t}\, dB_t^2$$

**Lecture :** le **clustering de volatilité** des modèles diffusifs (Heston) émerge comme limite macroscopique du **clustering d'événements** des modèles Hawkes microscopiques. C'est le premier pas vers un modèle unifié « across scales ».

**Hawkes marqués (Fauth & Tudor 2012).** Multiplication du noyau par une fonction du volume des trades : plus les volumes sont gros, plus les inter-arrivées sont courtes. Ajuste le modèle aux taux de change FX.

### 4.7 QHawkes — Hawkes quadratiques

**Motivation : effets de levier et de Zumbach.**

**Effet de levier** : rendements négatifs → augmentation de volatilité future. Formellement, pour $\tau > 0$ :
$$|\text{Cov}(dP_t, \sigma_{t-\tau}^2)| < |\text{Cov}(dP_{t-\tau}, \sigma_t^2)|$$
(corrélation **asymétrique** : le passé des prix influe plus sur la volatilité future que l'inverse). Intuition : la panique due aux chocs négatifs génère plus de volatilité, alors qu'une volatilité élevée ne prédit pas de mouvement de prix orienté.

**Effet de Zumbach** (asymétrie de la corrélation entre vol HF et vol LF) : la vol LF passée influe plus sur la vol HF future que l'inverse. Cov$(\sigma_t^2, R_{t,\tau}^2) < $ Cov$(\sigma_t^2, R_{t,-\tau}^2)$. Intuition : un marché tendanciel accroît l'incertitude et donc la volatilité future.

**Définition QHawkes (Blanc, Donier, Bouchaud).** $dP_t = \varepsilon_t \psi dN_t$ (prix qui font des sauts $\pm \psi$, $\psi$ = tick) et l'intensité :
$$\boxed{\lambda_t = \lambda_0 + \frac{1}{\psi}\int_{-\infty}^t L(t-s)\, dP_s + \frac{1}{\psi^2}\int_{-\infty}^t \int_{-\infty}^t Q(t-s, t-u)\, dP_s\, dP_u}$$

$L$ = noyau de **levier** (linéaire en $dP_s$), $Q$ = noyau **quadratique** (bilinéaire). Positivité requise.

**Cas particulier ZHawkes** (pour capturer Zumbach) : $L \equiv 0$, $Q(s, u) = \varphi(s)\delta(s-u) + z(s) z(u)$. L'intensité devient $\lambda_t = \lambda_0 + H_t + Z_t^2$ où $H_t$ est la partie Hawkes classique (diagonale) et $Z_t = \frac{1}{\psi} \int z(t-s) dP_s$ la partie tendance (rang 1).

### 4.8 Modèles d'ordre LOB

**Level-I de Large (2007).** Modèle 10D Hawkes : MOs et LOs qui bougent le prix (4 composantes), MOs et LOs qui laissent le prix inchangé (4 composantes), cancellations (2 composantes). Introduit le **response kernel** mesurant la résilience du LOB.

**Level-I de Bacry et al. (2016) — 8D.** $\mathbf{N}_t = (P^{(a)}, P^{(b)}, T^{(a)}, T^{(b)}, L^{(a)}, L^{(b)}, C^{(a)}, C^{(b)})$ : mouvements mid-price (ask/bid), trades non-mouvants (ask/bid), LOs non-mouvants (ask/bid), cancels (ask/bid).

**Résultats empiriques à citer :**
1. **Dynamique majoritairement auto-excitante** sauf pour les changements de mid-prix qui sont surtout cross-excités (matrice diagonale SAUF dans le bloc prix qui est anti-diagonal).
2. **Prix mainly triggered by prices** (effet anti-diagonal).
3. Marché **très endogène** (branching ratio proche de 1), noyaux en **loi de puissance** décroissance lente.
4. **Effets inhibiteurs** : certains noyaux sont négatifs. Ex : pour un actif large tick, un mouvement haussier inhibe les trades côté ask.

**Modèles d'ordre book complet.**
- **Muni Toke (2010)** : 2 agents, providers (LOs) et takers (MOs), volumes exponentiels, Hawkes exponentiels. Montre que les inter-temps MO-LO sont bien mieux modélisés par Hawkes 2D avec kernel cross (LM) qu'en Poisson.
- **Abergel-Lu (2018)** : 12 types d'événements (L/M/C × buy/sell × 0/1 selon impact mid-price). Compare Hawkes linéaires vs non-linéaires, QQ-plot + signature plot comme critères.

---

# PARTIE II — HFT3 : Approximation stochastique et exécution optimale

Le cours HFT3 est le pont entre théorie probabiliste (convergence d'algorithmes stochastiques) et pratique (exécution optimale d'ordres). **Thème unifiant** : beaucoup de problèmes de finance peuvent s'écrire $\text{trouver } \theta^* \text{ tel que } h(\theta^*) = 0$ où $h(\theta) = \mathbb{E}[H(\theta, Y)]$, et on ne peut évaluer $h$ que via des réalisations bruitées $H(\theta, Y)$.

## Chapitre 1 — Introduction à l'approximation stochastique

### 1.1 Motivations

Nombreux problèmes se ramènent à un **zero search** ou une **optimisation** d'une fonction $h$ qui n'est accessible que par simulation :
- **Finance** : extraction de paramètres implicites (volatilité), calibration, réduction de variance (importance sampling, stratification).
- **Économie** : minimisation de coûts, optimisation de profits.
- **Physique** : points d'équilibre d'un système (minimum d'un potentiel).
- **Statistique** : estimateurs MLE, moindres carrés, M-estimateurs.
- **Machine learning / réseaux de neurones** : recherche des poids minimisant la fonction de coût (descente de gradient stochastique, SGD).

**Point commun fondamental** :
$$h(\theta) = \mathbb{E}[H(\theta, Y)], \quad Y \text{ v.a. de dimension } q$$

L'approximation stochastique est l'outil basé sur la simulation qui résout $h(\theta) = 0$.

### 1.2 Cadre général (Robbins-Monro)

**Procédure récursive** sur un espace filtré $(\Omega, \mathcal{A}, (\mathcal{F}_n)_{n \geq 0}, \mathbb{P})$, à valeurs dans un ensemble convexe $\mathcal{C} \subset \mathbb{R}^d$ :
$$\boxed{\forall n \geq 0, \quad \theta_{n+1} = \theta_n - \gamma_{n+1} h(\theta_n) + \gamma_{n+1}(\Delta M_{n+1} + r_{n+1})}$$

avec :
- $(\gamma_n)_{n \geq 1}$ = **pas** (step sequence), valeurs dans $(0, \bar\gamma]$.
- $h : \mathcal{C} \to \mathbb{R}^d$ continue à croissance linéaire vérifiant $(\mathrm{Id} - \gamma h)(\mathcal{C}) \subset \mathcal{C}$ pour tout $\gamma \in (0, \bar\gamma]$ (stabilité).
- $\theta_0$ v.a. $\mathcal{F}_0$-mesurable finie.
- $\Delta M_n$ = **incrément de martingale** ($\mathcal{F}_n$-adapté, d'espérance conditionnelle nulle). C'est le « bruit » intrinsèque à la mesure.
- $r_n$ = **reste** (biais résiduel), $\mathcal{F}_n$-adapté.

**Intuition à maîtriser.** À chaque étape :
- On fait un petit pas de $-\gamma h(\theta_n)$ (descente type gradient vers la solution),
- Mais on reçoit une information **bruitée** ($\Delta M$ = bruit centré aléatoire) et potentiellement **biaisée** ($r$ = biais tendant vers 0).
- Le pas $\gamma_{n+1}$ diminue avec le temps pour amortir le bruit.

**Lien avec le SGD de ML** : c'est exactement la descente de gradient stochastique. $h(\theta) = \nabla f(\theta)$ où $f$ est la fonction de perte espérée, $\Delta M$ est l'erreur d'estimation du gradient sur un mini-batch, $r = 0$ typiquement.

### 1.3 Conditions fondamentales sur le pas

Les **conditions de Robbins-Monro** pour la convergence :
$$\boxed{\sum_{n \geq 1} \gamma_n = +\infty \quad \text{et} \quad \sum_{n \geq 1} \gamma_n^2 < +\infty}$$

**Pourquoi ces deux conditions ?**
- $\sum \gamma_n = \infty$ : l'algorithme doit pouvoir **parcourir toute la distance** nécessaire pour atteindre $\theta^*$ (sinon il s'arrête à mi-chemin). Comparable à l'intuition : la « somme des pas » doit pouvoir couvrir toute la carte.
- $\sum \gamma_n^2 < \infty$ : le **bruit cumulé** doit être contrôlable. Si les pas sont trop grands trop longtemps, le bruit fait osciller $\theta_n$ sans convergence. Comparable : il faut bien « amortir » à long terme.

**Choix canonique** : $\gamma_n = c/n$ (avec $c$ constante bien choisie). $\sum 1/n = \infty$, $\sum 1/n^2 < \infty$ ✓.

**Alternative** : $\gamma_n = c/n^\theta$ avec $\theta \in (1/2, 1]$ (plus lent, mais meilleure pour le moyennage — voir Ruppert-Polyak).

## Chapitre 2 — Convergence presque sûre

### 2.1 Approche martingale (fonction de Lyapunov)

**Hypothèses.** Il existe une fonction essentiellement quadratique $L : \mathbb{R}^d \to \mathbb{R}_+$ telle que :
- $\|h\|^2 \leq K(1 + L)$ (croissance contrôlée),
- $\forall \theta \neq \theta^*, \langle \nabla L | h \rangle(\theta) > 0$ (**condition de mean-reversion** vers $\theta^*$).

**Autres hypothèses** :
- $\theta_0 \in \mathcal{F}_0$ avec $L(\theta_0) < +\infty$ p.s.
- Pas $(\gamma_n)$ vérifiant Robbins-Monro.
- Bruit : $\mathbb{E}[\|\Delta M_{n+1}\|^2 | \mathcal{F}_n] \leq C(1 + L(\theta_n))$.
- Reste : $\sum_n \gamma_n \mathbb{E}[\|r_n\|^2 | \mathcal{F}_{n-1}] < \infty$.

**Conclusion** : $\theta_n \xrightarrow{p.s.} \theta^*$.

**Intuition — rôle de $L$ :**
- $L$ est une **fonction de Lyapunov** : une « énergie » qui doit décroître en moyenne le long de la trajectoire.
- La condition $\langle \nabla L | h \rangle > 0$ pour $\theta \neq \theta^*$ signifie : la direction de descente $-h$ pointe dans le sens de la décroissance de $L$. Autrement dit, suivre $-h$ revient à descendre la colline $L$ jusqu'au fond (où $\theta = \theta^*$).
- Exemple typique : $L(\theta) = \frac{1}{2}\|\theta - \theta^*\|^2$, et $h$ est un gradient.

### 2.2 Méthode de l'ODE

**Idée.** À l'échelle macroscopique (quand $n \to \infty$), la trajectoire $(\theta_n)$ « imite » celle de l'ODE déterministe :
$$\mathrm{ODE}_h \equiv \dot x = -h(x)$$

**Hypothèses principales** : l'ODE a un flot $\Phi(t, \xi)$ à valeurs dans $\mathcal{C}$ (vrai si $h$ est localement Lipschitz à croissance linéaire), $r_n \to 0$ p.s., bruit L2-borné, Robbins-Monro sur les pas.

**Théorème.** Sur l'événement $A_\infty = \{\omega : (h(\theta_n(\omega)))_n \text{ bornée}\}$, l'ensemble des points limites $\Theta_\infty(\omega)$ est un **attracteur minimal connexe compact flot-invariant** pour l'ODE.

Si en plus $\{h = 0\} = \{\theta^*\}$ et le flot $\Phi(\theta_0, t) \to \theta^*$ localement uniformément, alors $\theta_n \xrightarrow{p.s.} \theta^*$.

**Rappel de terminologie** :
- **Attracteur interne** $A$ de $K$ : $A \subsetneq K$ et $\forall x$ proche de $A$, $\text{dist}(\Phi(t,x), A) \to 0$.
- **Attracteur minimal** : pas d'attracteur interne strict.
- **Zéros de $h$ = attracteurs minimaux** par définition (peu importe leur stabilité).

**À l'examen, sache expliquer** : l'approche ODE ramène l'étude de l'algorithme stochastique à celle de la dynamique déterministe sous-jacente. Pour étudier si $\theta_n \to \theta^*$, on étudie $\dot x = -h(x)$ et on vérifie que $\theta^*$ est un attracteur global.

## Chapitre 3 — Vitesse(s) de convergence

### 3.1 TCL selon les valeurs propres

On suppose $h$ $\eta$-différentiable en $\theta^*$ :
$$h(\theta) = h(\theta^*) + J_h(\theta^*)(\theta - \theta^*) + o(\|\theta - \theta^*\|^{1+\eta})$$

Avec $\gamma_n = 1/n$ et le pas fixé, **3 régimes** selon la partie réelle de la plus petite valeur propre $\lambda_{\min}$ de $J_h(\theta^*)$ :

**Régime 1 : $\mathrm{Re}(\lambda_{\min}) > 1/2$** — le « régime standard » :
$$\sqrt{n}(\theta_n - \theta^*) \xrightarrow{\mathcal{L}-\text{stably}} \mathcal{N}(0, \Sigma^*), \qquad \Sigma^* = \int_0^{+\infty} e^{-u(J_h(\theta^*)^t - I_d/2)}\, \Gamma^*\, e^{-u(J_h(\theta^*) - I_d/2)}\, du$$

avec $\Gamma^* = \lim_n \mathbb{E}[\Delta M_{n+1}\Delta M_{n+1}^t | \mathcal{F}_n]$.

**Régime 2 : $\mathrm{Re}(\lambda_{\min}) = 1/2$** — cas limite, vitesse logarithmiquement ralentie :
$$\sqrt{\frac{n}{\log n}}(\theta_n - \theta^*) \xrightarrow{\mathcal{L}-\text{stably}} \mathcal{N}(0, \Sigma^*)$$

**Régime 3 : $\mathrm{Re}(\lambda_{\min}) = \Lambda \in (0, 1/2)$** — vitesse dégradée, ordre $n^{-\Lambda}$ au lieu de $n^{-1/2}$ :
$$n^\Lambda(\theta_n - \theta^*) \text{ p.s. borné}$$

Si $\Lambda = \lambda_{\min}$ (réel), cela converge vers une v.a. aléatoire (limite non gaussienne en général).

**Pourquoi cette dichotomie à 1/2 ?** Heuristique : $\sqrt{\gamma_n} \sim 1/\sqrt{n}$ donne la taille typique du bruit cumulé. Pour que ce bruit soit dominé par la contraction $-J_h(\theta^*)(\theta - \theta^*)$, il faut que la contraction soit assez rapide — et c'est elle qui est quantifiée par $\mathrm{Re}(\lambda_{\min})$. En dessous du seuil 1/2, le bruit « gagne » et l'algorithme converge plus lentement.

**Implication pratique.** Si on peut **choisir** le pas $\gamma_n = c/n$, il faut choisir $c$ assez grand pour que $c \cdot \mathrm{Re}(\lambda_{\min}) > 1/2$. Mais $c$ trop grand cause de l'instabilité numérique au début. D'où l'intérêt du **moyennage**.

### 3.2 Principe de moyennage Ruppert-Polyak

**Idée.** Utiliser un **pas plus grand** que $1/n$ (donc convergence plus rapide, mais plus bruitée), et « lisser » le résultat par moyenne arithmétique.

**Choix du pas** :
$$\gamma_n \sim \left(\frac{c}{b + n}\right)^\vartheta, \qquad \vartheta \in (1/2, 1)$$

**Procédure** :
$$\theta_{n+1} = \theta_n - \gamma_{n+1}(h(\theta_n) + \Delta M_{n+1})$$
$$\bar\theta_n = \frac{\theta_0 + \cdots + \theta_{n-1}}{n}$$

**Théorème (magie du moyennage).** Sous des hypothèses appropriées (dont la matrice $\text{Dh}(\theta^*)$ a des valeurs propres à parties réelles positives) :
$$\sqrt{n}(\bar\theta_n - \theta^*) \xrightarrow{\mathcal{L}} \mathcal{N}(0, \text{Dh}(\theta^*)^{-1} \Gamma \text{Dh}(\theta^*)^{-t})$$

**Point fondamental à retenir** : cette **variance asymptotique** est la **borne de Cramér-Rao** pour ce problème d'estimation. C'est-à-dire que l'estimateur moyenné $\bar\theta_n$ atteint **la meilleure variance possible** parmi toutes les procédures récursives. On a donc **optimalité asymptotique**, sans avoir besoin de régler précisément le pas $\gamma_n = 1/n$ avec la « bonne » constante.

**En pratique** : très utile car le choix des paramètres du pas est beaucoup plus robuste.

---

## Chapitre 4 — Application 1 : Répartition optimale d'ordres entre dark pools

### 4.1 Modélisation des dark pools

Un **dark pool** est un lieu d'exécution sans transparence pré-négociation :
- Propose un prix d'achat sans garantie de quantité exécutée (pour un OTC).
- Prix généralement inférieur au bid du marché régulier (pas d'impact de marché car pas de pre-trade transparency).

**Modèle formel** avec $N \geq 2$ dark pools :
- $V > 0$ : volume aléatoire à exécuter.
- $\theta_i \in (0, 1)$ : coefficient de discount proposé par le dark pool $i$.
- $r_i$ : pourcentage du volume envoyé au pool $i$ pour exécution.
- $D_i \geq 0$ : quantité effectivement disponible au pool $i$ au prix $\theta_i S$.

### 4.2 Fonction de coût

Le reste de l'ordre est exécuté sur le marché régulier au prix $S$. Coût total :
$$C = S \sum_{i=1}^N \theta_i \min(r_i V, D_i) + S\left(V - \sum_{i=1}^N \min(r_i V, D_i)\right) = S\left(V - \sum_{i=1}^N \rho_i \min(r_i V, D_i)\right)$$

avec $\rho_i = 1 - \theta_i \in (0, 1)$ = rabais.

### 4.3 Problème d'optimisation

**Minimiser le coût moyen** ⇔ **Maximiser** :
$$\max\left\{\sum_{i=1}^N \rho_i \mathbb{E}[S \min(r_i V, D_i)] : r \in \mathcal{P}_N\right\}$$

avec $\mathcal{P}_N = \{r = (r_i) \in \mathbb{R}_+^N : \sum r_i = 1\}$ (simplexe des allocations).

En notant $\widetilde V = V S$, $\widetilde D_i = D_i S$, on pose :
$$\varphi_i(u) = \rho_i \mathbb{E}[\min(u V, D_i)], \qquad \Phi(r_1, \ldots, r_N) = \sum_{i=1}^N \varphi_i(r_i)$$

### 4.4 Approche lagrangienne — conditions d'optimalité

Lagrangien pour la contrainte $\sum r_i = 1$ :
$$L(r, \lambda) = \Phi(r) - \lambda\left(\sum_{i=1}^N r_i - 1\right), \qquad \frac{\partial L}{\partial r_i} = \varphi'_i(r_i) - \lambda$$

Condition d'optimalité : **$\varphi'_i(r_i^*)$ est constant** sur $i$, ou de façon équivalente :
$$\varphi'_i(r_i^*) = \frac{1}{N}\sum_{j=1}^N \varphi'_j(r_j^*), \qquad \forall i \in I_N$$

### 4.5 Calcul du gradient et conception de l'algorithme

Clé : $\varphi'_i(r) = \rho_i \mathbb{E}[\mathbf{1}_{\{r_i V < D_i\}} V]$.

Donc, si $\arg\max_{\mathcal{H}_N} \Phi = \arg\max_{\mathcal{P}_N} \Phi \subset \mathrm{int}(\mathcal{P}_N)$ :
$$r^* \in \arg\max_{\mathcal{P}_N} \Phi \iff \mathbb{E}\!\left[V\!\left(\rho_i \mathbf{1}_{\{r_i^* V < D_i\}} - \frac{1}{N}\sum_{j=1}^N \rho_j \mathbf{1}_{\{r_j^* V < D_j\}}\right)\right] = 0, \quad \forall i$$

**Algorithme de recherche de zéro** (forme canonique RM) :
$$\boxed{r_i^{n+1} = r_i^n + \gamma_{n+1} H_i(r^n, Y^{n+1}), \quad r^0 \in \mathcal{P}_N}$$

avec :
$$H_i(r, Y) = V\left(\rho_i \mathbf{1}_{\{r_i V < D_i\}} - \frac{1}{N}\sum_{j=1}^N \rho_j \mathbf{1}_{\{r_j V < D_j\}}\right)$$

**Interprétation économique (importante !)** : l'algorithme **récompense les dark pools qui sur-performent la moyenne** (augmente leur allocation) et **pénalise ceux qui sous-performent** (diminue leur allocation). C'est une forme de renforcement par comparaison relative.

### 4.6 Problème de contrainte — $r_i > 0$

L'algorithme respecte $\sum r_i = 1$ par construction mais **pas** $r_i > 0$. Deux solutions :
1. **Fonction de Lyapunov + hypothèse de mean-reversion** hors du simplexe (plus simple math.).
2. **Troncature-projection** : forcer $r_i$ à rester dans $\mathcal{P}_N$ (plus efficace en pratique).

### 4.7 Tests numériques

Deux situations d'intérêt :
- **Abondance** : $\mathbb{E} V \leq \sum \mathbb{E} D_i$ (assez d'offre).
- **Pénurie** : $\mathbb{E} V > \sum \mathbb{E} D_i$ (demande excède l'offre).

**Benchmark « oracle »** : un insider connaît $V^n$ et $D_i^n$ avant chaque période et choisit la meilleure allocation. Il donne la réduction maximale de coût $\mathrm{CR}_{\text{oracle}}$. L'**indice de performance** de l'algo est alors :
$$\frac{\mathrm{CR}_{\text{opti}}}{\mathrm{CR}_{\text{oracle}}}$$

**Setting IID** (log-normales, $N=3$) : convergence rapide vers des ratios proches de 100%.

**Setting « pseudo-réel »** (données BNP, $N$ actifs corrélés) :
$$D_i = \beta_i\!\left[(1-\alpha_i) V + \alpha_i S_i \cdot \frac{\mathbb{E} V}{\mathbb{E} S_i}\right]$$

$\alpha_i$ = coefficients de recombination, $\beta_i$ = facteurs d'échelle. Pénurie = $\sum \beta_i < 1$. L'algo apprend à exploiter les corrélations entre actifs.

**Utilité pratique** : réinitialisation journalière possible, l'algo s'adapte au régime du jour tout en gardant un prior inter-jour.

---

## Chapitre 5 — Application 2 : Prix optimal de posting d'ordres limites

### 5.1 Modélisation du processus d'exécution

Sur une période courte $[0, T]$, on modélise l'exécution d'un **ordre d'achat** posté à $S_0 - \delta$ (à distance $\delta$ du prix initial $S_0$) par un **Poisson inhomogène** :
$$N_t^{(\delta)}, \quad 0 \leq t \leq T, \quad \text{d'intensité } \Lambda_T(\delta, S) = \int_0^T \lambda(S_t - (S_0 - \delta))\, dt$$

où :
- $\delta \in [0, \delta_{\max}]$ distance de posting ($\delta_{\max} \in (0, S_0)$ profondeur du carnet).
- $(S_t)$ processus stochastique du best ask.
- $\lambda$ fonction de la distance entre ask courant et prix postés ; le choix canonique est :
$$\boxed{\lambda(x) = A e^{-a x}}$$

**Interprétation** : plus l'ordre est posté profondément dans le carnet (ask – (S0-δ) grand, donc δ petit), moins il a de chance d'être exécuté. L'exponentielle est un ajustement empirique bien documenté (cf. Bouchaud et al.).

### 5.2 Estimation empirique de $\lambda$

Étape 1 : estimer $\lambda$ globalement. Soit $(\tau_n)$ les inter-arrivées de trades i.i.d. $\mathcal{E}(\lambda)$ :
$$\hat\lambda_n = \frac{\sum_{k=1}^n \mathbf{1}_{\{\tau_k < T\}}}{\sum_{k=1}^n \tau_k \wedge T}$$

Convergence p.s. vers $\lambda$ et TCL standard.

Étape 2 : paramétrer $\hat\lambda_n(\delta) = A e^{-a\delta}$ et estimer $A, a$ par régression linéaire sur $\ln \hat\lambda_n(\delta)$ vs $\delta$. Ex. sur Accor, janvier 2012 : $A_{\text{buy}} \approx 0.031$, $a_{\text{buy}} \approx 0.23$ (valeurs similaires côté sell, marché symétrique).

### 5.3 Fonction de coût d'exécution

**Scénario** :
- Portefeuille de taille $Q_T \in \mathbb{N}$ à exécuter sur $[0, T]$.
- À $t = 0$, on poste tout au prix $S_0 - \delta$.
- Sur $[t, t + \Delta t]$, proba d'exécution = $\lambda(S_t - (S_0 - \delta)) \Delta t$.
- Coût d'exécution de la quantité exécutée : $\mathbb{E}[(S_0 - \delta)(Q_T \wedge N_T^{(\delta)})]$.
- **Pénalisation** de la quantité non exécutée : on doit finir à $Q_T$. Soit $\Phi : \mathbb{R} \to \mathbb{R}_+$ convexe croissante avec $\Phi(0) = 0$.

**Coût d'exécution total** :
$$\boxed{C(\delta) = \mathbb{E}\!\left[(S_0 - \delta)(Q_T \wedge N_T^{(\delta)}) + \kappa S_T \Phi\!\left((Q_T - N_T^{(\delta)})_+\right)\right]}$$

$\kappa > 0$ = coefficient de pénalité de market impact pour la quantité résiduelle.

**Problème** : $\min_{0 \leq \delta \leq \delta_{\max}} C(\delta)$.

**Interprétation à retenir** : c'est un arbitrage classique en exécution optimale :
- **$\delta$ grand** (on poste profond dans le carnet) : prix avantageux, mais **grande probabilité de non-exécution** → forte pénalité.
- **$\delta$ petit** (on poste près du best ask) : prix moins avantageux, mais **forte probabilité d'exécution** → faible pénalité.

Il existe un $\delta^*$ optimal qui équilibre les deux.

### 5.4 Stratégie : représentation de $C'$ comme espérance

Pour appliquer l'approximation stochastique, on a besoin :
1. D'une représentation $C'(\delta) = \mathbb{E}[H(\delta, S)]$ avec $H$ calculable.
2. De conditions assurant convexité stricte et $C'(0) < 0$ (sinon $\delta^* = 0$ trivialement).

**Faits de base sur le Poisson** (indispensables pour dériver sous l'espérance) : pour $(N^\mu)$ famille Poisson de paramètre $\mu$,
- $\frac{d}{d\mu}\mathbb{E}[f(N^\mu)] = \mathbb{E}[f(N^\mu)(N^\mu/\mu - 1)]$ (pour $f$ à croissance polynomiale).
- $\mathbb{E}[k \wedge N^\mu] = k \mathbb{P}(N^\mu > k) + \mu \mathbb{P}(N^\mu \leq k-1)$.
- $\mathbb{E}[(k - N^\mu)_+] = k \mathbb{P}(N^\mu \leq k) - \mu \mathbb{P}(N^\mu \leq k-1)$.

**Représentations** (avec notations $\mathbb{P}^{(\delta)}$, $\mathbb{E}^{(\delta)}$ = substitution $\mu = \Lambda_T(\delta, S)$) :

**Cas $\Phi \neq \mathrm{id}$** :
$$H(\delta, S) = -Q_T \mathbb{P}^{(\delta)}(N^\mu > Q_T) - \kappa S_T \frac{\partial}{\partial \delta}\Lambda_T(\delta, S) \cdot \varphi^{(\delta)}(\mu) + \left[\frac{\partial}{\partial \delta}\Lambda_T(\delta, S)(S_0 - \delta) - \Lambda_T(\delta, S)\right]\mathbb{P}^{(\delta)}(N^\mu \leq Q_T - 1)$$

avec $\varphi^{(\delta)}(\mu) = \mathbb{E}^{(\delta)}[(\Phi(Q_T - N^\mu) - \Phi(Q_T - N^\mu - 1))\mathbf{1}_{\{N^\mu \leq Q_T - 1\}}]$.

**Cas $\Phi = \mathrm{id}$** (simplification) :
$$H(\delta, S) = -Q_T \mathbb{P}^{(\delta)}(N^\mu > Q_T) + \left[(S_0 - \delta - \kappa S_T)\frac{\partial}{\partial \delta}\Lambda_T(\delta, S) - \Lambda_T(\delta, S)\right]\mathbb{P}^{(\delta)}(N^\mu \leq Q_T - 1)$$

### 5.5 Critères de monotonie à l'origine et de convexité

Supposons $(S_t)$ vérifiant un **principe de co-monotonie fonctionnelle** (propriété technique satisfaite par le brownien, Lévy, etc.). Avec $\lambda(x) = A e^{-ax}$ :

**(a) Monotonie à l'origine** : $C'(0) < 0$ (sinon pas de gain à déposer avec $\delta > 0$) ssi :
$$Q_T \geq 2 T \lambda(-S_0) \quad \text{et} \quad \kappa \leq \frac{1 + a S_0}{a\mathbb{E}[S_T](\Phi(Q_T) - \Phi(Q_T - 1))}$$

Dans le cas $\Phi \equiv \mathrm{id}$ : $\kappa \leq \frac{1 + a S_0}{a\mathbb{E}[S_T]}$.

**(b) Convexité** : $C''(\delta) \geq 0$ ssi (si $\Phi \neq \mathrm{id}$, satisfaisant $\Phi(x) - \Phi(x-1) \leq \rho_Q(\Phi(x+1) - \Phi(x))$ pour $\rho_Q \in (0,1)$) :
$$Q_T \geq 2 T \lambda(-S_0) \quad \text{et} \quad \kappa \leq \frac{2}{a\mathbb{E}[S_T]\Phi'_\ell(Q_T)}$$

**Signification** : il faut que (i) le portefeuille soit assez gros, (ii) la pénalisation ne soit pas trop violente. Si ces deux conditions sont remplies, le problème est bien posé (convexe avec solution intérieure unique).

### 5.6 Design de l'algorithme

Procédure d'approximation stochastique **avec projection** :
$$\boxed{\delta_{n+1} = \mathrm{Proj}_{[0, \delta_{\max}]}\left(\delta_n - \gamma_{n+1} H(\delta_n, \bar S_{t_i}^{(n+1)})_{0 \leq i \leq m}\right)}$$

- $\mathrm{Proj}$ = projection euclidienne sur $[0, \delta_{\max}]$.
- $(\gamma_n)$ : décroissance adéquate.
- $(\bar S_{t_i}^{(n)})_n$ : suite i.i.d. de copies de la dynamique $(S_{t_i})$ (ou schéma d'Euler, ou séquence α-mixing).

### 5.7 Expériences numériques

**Données simulées** : $dS_t = \sigma dW_t$, schéma d'Euler avec $m = 20$ pas, $M = 10000$ simulations MC. Paramètres typiques : $s_0 = 100, \sigma = 0.005, A = 5, a = 1, T = 5, Q = 10$, $\kappa = 6$, $A' = 1, a' = 0.01$ (pour $\Phi(x) = (1 + A' e^{a'x}) x$).

**Données marché** (Accor, 11/11/2010) : on divise la journée en cycles de 15 trades, $N_{\text{cycles}} = 220$. L'intensité empirique est estimée sur ces cycles. Algorithme avec $\gamma_n = 1/(550 n)$ (vs algorithme moyenné Ruppert-Polyak $\gamma_n = 1/(550 n^{0.95})$).

**Résultat-clé** : Ruppert-Polyak donne une estimation de $\delta^*$ et du prix de posting plus **stable** que l'algorithme nu. À l'examen : cite ce résultat pour justifier l'usage du moyennage.

---

# PARTIE III — HFT2 : Faits stylisés (II) et Estimation HF

Le cours HFT2 a quatre grands chapitres : (1) relations entre indicateurs quotidiens, (2) **estimation de la volatilité et de la corrélation en microstructure** (bloc central demandé par l'énoncé), (3) algorithmes de trading, (4) le Flash Crash du 6 mai 2010.

## Chapitre 1 — Relations entre indicateurs quotidiens

### 1.1 Volume quotidien vs nombre de trades

Empiriquement (NYSE, Euronext-Paris 2002-2011), $\bar V_d$ (volume moyen quotidien) et $\bar N_d$ (nombre moyen de trades) sont **fortement corrélés positivement**. Mais le ratio $\bar V_d / \bar N_d$ (taille moyenne par trade) **a diminué** dans le temps, reflétant l'essor du **slicing** : les ordres sont coupés en plus petits morceaux, donc plus de trades pour un volume donné.

**À l'examen** : expliquer que la structure de la relation volume/trades est le produit d'une évolution technologique + réglementaire (MiFID 2007, Reg NMS 2007) qui a favorisé les algos d'exécution et le HFT.

### 1.2 Quantité au best quote vs volume négocié

La **quantité aux meilleures limites** $\bar Q_d$ est une mesure de liquidité. Empiriquement, elle croît avec le volume négocié — mais la relation dépend de la **taille du tick** :
- **Tick large** : le spread est toujours de 1 tick, donc le prix moyen d'un trade est fixé par le tick et la probabilité de vider une file.
- **Tick petit** : chaque trade déclenche un changement de prix ; le prix moyen d'un trade est fixé par le spread moyen.

### 1.3 Volatilité vs spread (modèle MRR)

**Modèle de Madhavan-Richardson-Roomans** (le plus simple, market-makers à l'équilibre) :
$$\boxed{\varphi = c \sigma_{\text{mid}} N^{-1/2}}$$

avec $\varphi$ = spread, $\sigma_{\text{mid}}$ = volatilité du mid-price, $N$ = nombre de trades, $c > 0$.

**Interprétation économique (clé) :**
- $\sigma_{\text{mid}}$ mesure l'**adverse selection** auquel font face les market-makers.
- $\varphi$ est leur **profit par trade**.
- $N^{-1/2}$ : par le TCL, la volatilité par trade décroît comme $1/\sqrt{N}$.
- À l'équilibre, le profit doit compenser l'adverse selection : $\varphi \propto \sigma / \sqrt{N}$.

**Autrement dit** : **spread ∝ volatilité par trade**. Si $\sigma$ double, le spread double (marché petit tick). Si $N$ double, le spread est divisé par $\sqrt 2$ (marché plus actif).

### 1.4 Échec du MRR pour les gros ticks

Pour les **large tick stocks** (ex. HKE, TSE), la relation $\varphi \propto \sigma_{\text{mid}}/\sqrt N$ **échoue**.

**Raison mécanique** : le spread ne peut pas descendre sous 1 tick. Quand la vol continue à baisser, le spread reste bloqué à 1 tick.

**Conséquence** : les market-makers tirent un **profit supplémentaire** dû à cette contrainte de liquidité (le spread minimal est $\geq$ spread d'équilibre théorique).

### 1.5 Approche de Dayri-Rosenbaum (spread implicite)

Pour étendre la relation au régime gros tick, Dayri-Rosenbaum utilisent le **modèle à zones d'incertitude** de Robert-Rosenbaum. Le tick $\alpha$ confine le prix continu « vrai » sur une grille discrète. On définit des **zones d'incertitude** de largeur $2\eta\alpha$ autour des milieux de ticks, avec $0 < \eta < 1$.

Interprétation : pour $\eta < 1/2$, prix en *mean-reversion microscopique* ; $\eta > 1/2$, prix en *tendance* ; $\eta = 1/2$, pas d'effet. La relation généralisée entre spread et volatilité par trade fait intervenir ce paramètre $\eta$.

## Chapitre 2 — Estimation de la volatilité et corrélation en microstructure

**(Chapitre central du cours — à maîtriser absolument.)**

### 2.1 Variance réalisée — l'estimateur naïf

Pour un prix $S$ et un pas $\Delta$ :
$$\boxed{\widehat V_R(\Delta) = \sum_{j=1}^{T/\Delta} (\ln S_{\Delta j} - \ln S_{\Delta(j-1)})^2 \xrightarrow[\Delta \to 0]{\mathbb P} \int_0^T \sigma_s^2\, ds}$$

**Justification** : si $dS = \sigma_t\, S\, dW_t$, alors $d\ln S = (\mu - \sigma^2/2)dt + \sigma dW$ et la variation quadratique de $\ln S$ est $\int_0^T \sigma_s^2 ds$. La somme des $(\Delta \ln S)^2$ converge vers la **variation quadratique**.

### 2.2 Problème du « signature plot »

**Observation empirique choquante** (Rio Tinto 2012, et partout ailleurs) : quand $\Delta \to 0$, au lieu de se stabiliser, $\widehat V_R(\Delta)$ **diverge** vers l'infini !

**Raison : le bruit de microstructure.** Le prix observé $X$ n'est pas la vraie semi-martingale $M$ (prix efficient), mais $X = M + \varepsilon$ avec $\varepsilon$ un bruit (bid-ask bounce, tick discret, asynchronisme, etc.).

$$r_\Delta(j) = X_{\Delta j} - X_{\Delta(j-1)} = \underbrace{M_{\Delta j} - M_{\Delta(j-1)}}_{r_\Delta^M(j)} + \underbrace{\varepsilon_{\Delta j} - \varepsilon_{\Delta(j-1)}}_{\eta_\Delta(j)}$$

Donc la variance réalisée vaut :
$$\widehat V_R(\Delta) = \sum_j r_\Delta(j)^2 = \underbrace{\sum_j r_\Delta^M(j)^2}_{\to \int \sigma^2 ds} + \underbrace{\sum_j \eta_\Delta(j)^2}_{\to \infty \text{ si bruit i.i.d.}} + 2 \sum_j r_\Delta^M(j)\eta_\Delta(j)$$

**Explication du terme qui diverge** : $\eta_\Delta(j) = \varepsilon_{\Delta j} - \varepsilon_{\Delta(j-1)}$ est d'ordre $O(1)$ (pas $O(\sqrt \Delta)$), donc $\sum \eta_\Delta(j)^2 \sim T/\Delta \to \infty$ quand $\Delta \to 0$.

**Morale pour l'examen** : la variance réalisée naïve **ne fonctionne pas** en très haute fréquence. Il faut soit (i) prendre $\Delta$ modérément grand (5 min classique — mais on perd de l'info), (ii) utiliser un estimateur corrigé.

### 2.3 Estimateur de Garman-Klass (OHLC)

Idée : exploiter les prix **open, close, high, low** de la période.

**Modèle** : $P_t = \Phi(X_t)$ avec $\Phi$ monotone, $dX_t = \sigma\, dW_t$, $\sigma$ constant à estimer.

**Estimateur GK** — obtenu comme « meilleur » estimateur quadratique, scale-invariant, sans biais, à variance minimale :
$$\boxed{\sigma_{GK}^2 = \frac{1}{2}(\max S - \min S)^2 - (2\ln 2 - 1)(S_{\text{end}} - S_{\text{begin}})^2}$$

**Lecture** : le premier terme utilise l'étendue (max-min) sur la période, le second corrige par la différence open-close. Coefficient $2 \ln 2 - 1 \approx 0.386$.

**Pourquoi c'est mieux** : les prix extrêmes (high, low) incorporent beaucoup d'information sur la volatilité — plus que juste deux points aléatoires. Estimateur plus efficace que le simple $(S_{\text{end}} - S_{\text{begin}})^2$.

**Application empirique** (Alstom 2011) : volatilité GK calculée heure par heure, forme en U (plus forte à l'ouverture et à la clôture).

### 2.4 Autres estimateurs de volatilité à mentionner

- **Multi-scale** (Aït-Sahalia, Aït-Sahalia-Jacod) : combine estimateurs à plusieurs échelles pour canceller le biais de bruit.
- **Subsampling** (Zhang, Mykland, Aït-Sahalia, ZMA) : moyenne sur plusieurs grilles décalées.
- **Two-scales realized volatility (TSRV)** : combinaison pondérée qui annule explicitement le biais du bruit.
- **Pré-averaging** (Jacod et al.) : on fait la moyenne mobile des prix avant de calculer les increments.
- **Bid-ask modeling** (Robert-Rosenbaum) : utilise la structure microscopique bid-ask elle-même.

## Chapitre 3 — Estimation de la covariance et corrélation HF

### 3.1 Estimateur classique

Deux prix $X^1 = \ln P^1$, $X^2 = \ln P^2$ avec $d\langle W^1, W^2\rangle_t = \rho_t\, dt$ :
$$dX^k = \mu_t^k\, dt + \sigma_{t-}^k\, dW_t^k, \quad k = 1, 2$$

Pour $r_\Delta^k(j) = X_{\Delta j}^k - X_{\Delta(j-1)}^k$ :
$$\int_0^T \rho_s \sigma_s^1 \sigma_s^2\, ds = \lim_{\Delta \to 0} \sum_{j=1}^{T/\Delta} r_\Delta^1(j)\, r_\Delta^2(j)$$

D'où l'**estimateur de covariance réalisée** :
$$\widehat C_R(\Delta) = \sum_{j=1}^{T/\Delta} r_\Delta^1(j)\, r_\Delta^2(j) \xrightarrow[\Delta \to 0]{\mathbb P} \int_0^T \rho_s \sigma_s^1 \sigma_s^2\, ds$$

**Problème majeur** : cela **requiert des données synchrones**. Or les trades et quotes sont asynchrones (chaque actif a ses propres temps d'arrivée).

### 3.2 Effet Epps

**Epps (1979)** : *"Correlations among price changes are found to decrease with the length of the interval for which the price changes are measured."*

Autrement dit, quand $\Delta \to 0$, la **corrélation empirique tend vers 0**, même si la corrélation théorique (entre les processus sous-jacents) est non-nulle.

**Explications proposées :**
1. **Biais systématique** de l'estimateur.
2. Effet **lead-lag** entre actifs du même secteur.
3. **Asynchronie des trades**.
4. Effets **tick** et microstructure.

**Exemple formel** : si $X^1, X^2$ sont deux browniens corrélés de corrélation $\rho$, mais observés aux temps d'arrivée de deux Poisson **indépendants**, alors $\mathbb{E}[\bar C_R(\Delta)] \to 0$ quand $\Delta \to 0$. C'est le **problème** que Hayashi-Yoshida résolvent.

### 3.3 Estimateur de Hayashi-Yoshida

**Construction.** Soient $I_i^1 = [T_i^1, T_{i+1}^1)$ et $I_j^2 = [T_j^2, T_{j+1}^2)$ les intervalles entre observations des deux actifs.

**Estimateur de covariance cumulée** :
$$\boxed{U_n = \sum_{i, j} \Delta P^1(I_i^1)\, \Delta P^2(I_j^2)\, \mathbf{1}_{\{I_i^1 \cap I_j^2 \neq \emptyset\}}}$$

**Interprétation en une phrase** : on fait le produit des increments **uniquement pour les couples d'intervalles qui se chevauchent temporellement**. Les autres sont ignorés.

**Deux estimateurs de corrélation associés** :

Si $\sigma_1, \sigma_2$ connues :
$$R_n^1 = \frac{1}{T} \frac{\sum_{i,j} \Delta P^1(I_i^1) \Delta P^2(I_j^2) \mathbf{1}_{\{I_i^1 \cap I_j^2 \neq \emptyset\}}}{\sigma_1 \sigma_2}$$

Si $\sigma_1, \sigma_2$ connues ou non :
$$R_n^2 = \frac{\sum_{i,j} \Delta P^1(I_i^1) \Delta P^2(I_j^2) \mathbf{1}_{\{I_i^1 \cap I_j^2 \neq \emptyset\}}}{\sqrt{\sum_i \Delta P^1(I_i^1)^2}\, \sqrt{\sum_j \Delta P^2(I_j^2)^2}}$$

**Propriétés importantes** :
1. **Pas de choix de $\Delta$** : on utilise les temps d'arrivée naturels. Gain massif en robustesse.
2. **Convergent** vers $\rho$ quand $n \to \infty$, sous l'hypothèse que les temps d'arrivée sont indépendants du prix (Hayashi-Yoshida 2005).
3. **Non robuste au bruit de microstructure** (comme l'estimateur naïf, mais sans le problème d'asynchronie).

**À l'examen** : insister sur l'idée géniale — utiliser l'indicatrice de chevauchement résout l'asynchronie sans ré-échantillonner (ce qui jette de l'information).

## Chapitre 4 — Algorithmes de trading

### 4.1 Types d'ordres (rappel microstructure)

- **Market-to-Limit (MtoL)** : LO envoyé à un prix qui croise le spread. Consomme la liquidité au best ask, le reste non exécuté reste posté comme nouveau best bid. *Risque : exécution partielle.*
- **Immediate-or-Cancel (IoC) / Fill-and-Kill (FaK)** : MtoL, mais ce qui n'est pas exécuté immédiatement est annulé. *Risque : exécution partielle.*
- **Fill-or-Kill (FoK)** : MtoL tout-ou-rien. Si pas entièrement exécutable, annulé. *FoK = FaK 100%.*
- **Peg order** : LO qui reste toujours au best (on annule et replace à chaque amélioration de la file).
- **Iceberg** : ordre dont une partie est visible, le reste caché. Quand la partie visible est exécutée, une nouvelle tranche apparaît. *Cache la taille réelle pour éviter l'impact informationnel.*
- **Stop-loss / Take-profit** : ordres de gestion de risque (sortir si la perte/gain atteint un seuil).

### 4.2 Slicing (découpage d'ordres)

**Problème** : quand un ordre représente une part significative du volume journalier, l'exécuter d'un coup produirait un impact de marché énorme.

**Solutions** :
1. **Block order** : via un broker qui trouve une contrepartie. *Risque : information révélée au broker et au marché ex-post.*
2. **Slice and dice** : découper en ordres enfants. *Avantage : information cachée au marché. Inconvénient : gérer le trading soi-même.*

### 4.3 TWAP — Time-Weighted Average Price

**Définition** : diviser la période d'exécution en $N$ tranches égales, envoyer la même taille à chaque tranche.

Ex. : 20 000 actions entre 9h et 17h30 ⇒ 51 périodes de 10 min, ~392 actions par période.

**Propriétés** :
- **Extrêmement prévisible** : toujours la même taille, même fréquence.
- **Détectable** par le marché. On peut rendre moins prévisible avec :
  - Taille aléatoire : $392 \pm \varepsilon$ actions.
  - Durée aléatoire : $10 \pm \tau$ minutes.

### 4.4 Percentage of Volume (PoV)

**Définition** : trader un pourcentage fixe du volume du marché sur chaque tranche.

Ex. : 50% de participation, 20 000 actions à acheter, volume du marché estimé sur les 30 derniers jours.

**Propriétés** :
- Moins prévisible que TWAP.
- **Dépendant de la courbe intraday** de volume (U-shape typique).
- **Statique** : nombre de shares fixé ex ante.
- **Adaptation dynamique** possible via prix cibles :
  - Range de participation $50 \pm \varepsilon$%.
  - Bornes de prix $p_0 \pm x$ EUR.
  - Ex : si $p \in [98, 102]$, 50% ; si $p < 98$, 75% ; si $p > 102$, 25%.
- **PoV dynamique** : risque d'exécution incomplète.

### 4.5 VWAP — Volume-Weighted Average Price

**Définition** : allouer les shares proportionnellement à la courbe de volume intraday.

$$w_n = \frac{V_n}{\sum_{n=1}^N V_n}, \qquad v_n = Q \cdot w_n$$

Ex. : si 10% du volume journalier est tradé entre 15h et 15h10, on achète 10% des 20 000 shares sur cette tranche = 2000 shares.

**Propriétés** :
- Moins prévisible que TWAP.
- Très dépendante de la courbe de volume intraday.
- Statique.
- Plus **flexible que PoV** : le temps final $N$ de l'exécution est un paramètre qu'on peut ajuster.

### 4.6 Implementation Shortfall (IS)

**Benchmark** : le prix au moment où l'exécution **commence** ($S_0$). On mesure la perte par rapport à $S_0$.

Méthode : **Almgren-Chriss** (contrôle optimal stochastique, au programme de cours sur l'exécution optimale).

### 4.7 Target Close

**Benchmark** : le prix au moment où l'exécution **se termine** (prix de clôture). Utilisé quand on veut réplicer le prix de clôture.

### 4.8 Récap comparatif (à savoir par cœur)

| Algo | Benchmark | Prévisibilité | Dép. courbe volume | Dynamique | Risque exéc. incomplète |
|------|-----------|---------------|---------------------|-----------|--------------------------|
| TWAP | Prix moyen sur [0,T] | Très forte | Non | Non | Non |
| PoV | Prix selon participation | Moyenne | Oui | Peut être | Oui (dynamique) |
| VWAP | Prix moyen pondéré volume | Moyenne | Oui (forte) | Peut être | Oui (dynamique) |
| IS | Prix au début | Faible | Oui | Oui | Non |
| Target Close | Prix à la fin | Faible | Oui | Oui | Non |

## Chapitre 5 — Flash Crash du 6 mai 2010

### 5.1 Le crash

- **Durée** : 36 minutes seulement.
- **Impact** : perte de **862 milliards USD** de capitalisation boursière.
- **Ampleur** : le Dow Jones a perdu quasiment **1000 points (~9%)** puis a récupéré quelques minutes plus tard.

### 5.2 Déclencheur

Un algorithme d'exécution $X$ d'un fonds traditionnel a lancé une **vente de 75 000 contrats E-Mini** (S&P 500 futures sur CME), ~4,1 milliards USD.

$X$ était de type **PoV à 9%** de la volumétrie de la dernière minute → **aucun contrôle sur le prix, le temps, ou l'impact marché**.

Normalement, une exécution de cette taille prend **au moins 5 heures**. $X$ l'a exécutée en **19 minutes** → 15× plus vite.

### 5.3 Déroulement

- 14h32 : premier ordre enfant de $X$ hit le marché.
- $X$ consomme la liquidité rapidement → HFTs et arbitragistes détectent et fournissent plus de liquidité.
- **Boucle infernale** : plus la liquidité est offerte, plus $X$ accélère (PoV sans plafond).
- La liquidité au bid ne suffit plus à contrer la pression vendeuse.
- 14h41-14h44 : les HFTs **capitulent** et commencent à vendre aussi (pour réduire l'inventaire accumulé). 2000 contrats vendus agressivement.
- HFTs non seulement cessent de fournir la liquidité, **ils la consomment**.
- **Réaction en chaîne** : liquidité au bid disparaît, chute libre du prix.

### 5.4 Leçons

- Un algorithme naïf (PoV sans plafond) couplé à une dynamique HFT peut créer une **crise de liquidité** en quelques minutes.
- Depuis : **circuit breakers** (interruptions automatiques si le prix dépasse certains seuils), meilleure régulation des algos.
- Importance des **paramètres de sécurité** dans tout algo d'exécution : prix plancher, cap de participation, limite de volume par période, etc.

---

# PARTIE IV — HFT1 : Faits stylisés (I) — Fondements de la microstructure

Le cours HFT1 pose les bases. Il comporte quatre chapitres : (1) histoire des marchés, (2) zoologie des acteurs, (3) marchés électroniques, (4) **indicateurs intraday** (bloc crucial demandé par l'énoncé : évolution intraday du volume, volatilité, spread).

## Chapitre 1 — Histoire des marchés

### 1.1 Définition d'un marché

Un **marché** est le lieu où acheteurs et vendeurs se rencontrent. Deux types :

**Exchange** (marché centralisé, type Bourse) :
- Instruments listés / standardisés (ex. futures, options listées).
- **Chambre de compensation** (clearing house) : marges de clearing + règlements quotidiens.
- ⇒ **Pas de risque de contrepartie / de crédit**.

**OTC (over-the-counter)** :
- N'importe quel instrument imaginable (forwards, options exotiques).
- Pas de garantie que les parties honorent le contrat.
- ⇒ **Risque de contrepartie / de crédit**.

### 1.2 De la criée à l'électronique

- **La Criée** : 1774-1987 à Paris. Négociation orale face-à-face.
- **Ticker Tape** (1870-1970) : premier moyen électrique de diffusion de prix (télégraphe).
- **NASDAQ** (1971) : premier système de cotation électronique.
- **NYSE DOT** (1970s) : système de routage électronique vers le floor.
- **CATS** (Toronto, 1977) : premier système automatique complet.
- **MPDS** (Londres, 1970) : premier système standardisé de diffusion de prix.

### 1.3 Histoire en Europe

- **SEAQ** (LSE, 1986) : diffusion d'info avant cotation électronique (trading par téléphone).
- **SAEF** (UK, 1988) : petit ordres automatisés.
- **CATS → CAC** (France, 1989) : Cotation Assistée en Continu.

### 1.4 MiFID (Markets in Financial Instruments Directive)

**Régulation européenne** augmentant la transparence et la concurrence.

**Entités couvertes** : firmes fournissant des « services d'investissement et activités ».

**Substance** :
1. **Autorisation home-state + passport** : autorisation dans le pays d'origine, puis libre circulation dans toute l'UE.
2. **Catégorisation clients** : retail / professional / eligible counterparty.
3. **Order handling** : exigences sur le traitement des ordres clients.
4. **Pre-trade transparency** : les opérateurs de systèmes continus de matching doivent publier les meilleurs prix / volumes agrégés.
5. **Post-trade transparency** : publication prix / volume / heure des trades.
6. **Best execution** : les firmes doivent prendre toutes les mesures raisonnables pour obtenir le meilleur résultat pour le client.
7. **Systematic Internalisers (SI)** : firmes traitant leurs propres clients en interne sont traitées comme mini-exchanges.

**MiFID II** (3 janvier 2018) : version révisée, notamment avec règles sur le HFT, tick size regime, transparence renforcée.

### 1.5 Reg NMS (US, 2007)

Équivalent américain : **National Market System**.
- Meilleure exécution forcée entre venues.
- Forte compétition → essor des HFT et fragmentation.

### 1.6 Fragmentation

**Avant MiFID/Reg NMS** : un seul exchange par pays dominant (Euronext pour Paris, LSE pour Londres, NYSE pour US).

**Après** : éclatement entre **lit venues** (exchanges), **MTFs** (Multilateral Trading Facilities, type Chi-X, BATS, Turquoise), **dark pools**, **SI** (Systematic Internalisers).

**Exemple CAC40 2008 → 2012** :
- 2008 : Euronext 87.7% + concurrents <12%.
- 2012 : Euronext 65% + BATS/Chi-X/Turquoise ~30%.

**Exemple FTSE100 2008 → 2012** :
- 2008 : LSE 75.6%.
- 2012 : LSE 54%, Dark MTFs 3.4%, SIs 2.7%, OTC 39.9%.

**Japon (Nikkei 225 2010-2011)** : passage d'un monopole TSE (92%) à une fragmentation rapide (TSE 87%, plus 4 concurrents).

**Conséquence** : l'information de marché s'éclate sur plusieurs venues. Il faut des algorithmes de **Smart Order Routing (SOR)** pour chercher la meilleure exécution.

## Chapitre 2 — Zoologie des acteurs

### 2.1 Métaphore biologique

Le marché = écosystème.
- Agents = espèces avec leur niche.
- Infrastructure + régulation + opportunités + agents = écosystème.
- Relations coopératives / compétitives / parasites.
- Mutation / sélection naturelle : stratégies évoluent, meilleures survivent.
- Invasion / extinction : quand règles changent ou nouveaux joueurs arrivent.

### 2.2 Les acteurs principaux

**Banques de détail** : rémunèrent les dépôts à bas taux, prêtent à taux élevé, profit = différence. Gestion du risque = solvabilité (*dull banking*).

**Banques d'investissement** : vendent des produits financiers aux investisseurs (sell-side). Créent nouveaux produits à la demande. Profit via volume + marge/prime sur le prix de couverture (*flow trading*). Produits complexes → quants.

**Brokers** : intermédiaires. Reliés entre investisseurs et marchés. Frais : fixes (gestion de compte) + par contrat tradé. Réputation = garantie. Profit = volume.

**Market-makers** : achètent et vendent en continu.
- Font des **quotes fermes** (prix + volume) : si contrepartie acceptante, obligés de trader.
- ⇒ **Liquidity providers**.
- Prix bid < prix ask → gagnent le **spread** sur chaque trade aller-retour.

**Petits épargnants** : dépôts en banque à un taux d'intérêt. Si plus confiance → retrait → *bank run* possible.

**Petits investisseurs** : < 100K USD. Pas d'accès direct à tous les services, passent par intermédiaires. Désavantage informationnel → plus de protection légale.

**Investisseurs qualifiés** : > 250K USD. Supposés aussi sophistiqués qu'un provider de services financiers. Moins de protection légale. Accès à des stratégies complexes (hedge funds, etc.). Horizon court/moyen terme (< 5 ans).

**Investisseurs institutionnels** : capital > 100M USD (fonds pension, souverains, (ré)assureurs). Accès à toutes les classes d'actifs. Contraintes légales/internes strictes. Horizon long terme, rendements stables.

**HFT firms** : acteurs récents (depuis 2000s). Stratégies automatisées à très haute fréquence. Market-making + arbitrage + liquidity-detection. Profit très petit par trade × énorme volume.

## Chapitre 3 — Marchés électroniques

### 3.1 Limit Order et Market Order

**Deux types d'ordres** :
- **Limit Order (LO)** : quote d'un fournisseur de liquidité. Price + quantity + side (buy/sell). Assis passivement dans le carnet en attendant un match.
- **Market Order (MO)** : consommation de liquidité. Exécution immédiate au meilleur prix opposé disponible (best ask pour acheter, best bid pour vendre). Hit l'ask ou le bid.

### 3.2 Limit Order Book (LOB)

Le **LOB** est l'offre de liquidité courante du marché à chaque niveau de prix. C'est l'agrégation de tous les LOs en attente.

**Deux côtés** :
- **Bid side** : prix auxquels le marché achète (ordres d'achat posés).
- **Ask side** : prix auxquels le marché vend (ordres de vente posés).

**Best bid** = plus haut prix d'achat. **Best ask** = plus bas prix de vente. **Best bid < Best ask**.

**Anonymat** : en électronique, les dealers sont anonymes. Données agrégées : on voit le volume total à chaque prix, pas qui a posté quoi. (Les exchanges ont les IDs pour la régulation.)

### 3.3 Microstructure du LOB

**Priorité** : typiquement **prix/temps**.
- **Prix** : les LOs au meilleur prix sont sélectionnés en premier.
- **Temps** : à prix égal, le plus ancien est sélectionné en premier (FIFO).

**Tick** : plus petit incrément de prix. Les prix sont discrets.
- US : 0.01 USD pour tous les stocks.
- Europe : variable (0.01 EUR en [50-99.99], 0.05 EUR au-dessus de 100, dépend aussi de la liquidité depuis MiFID II).

**Spread** = best ask − best bid. **Interprétations** :
- **Prix de la liquidité** : gain attendu d'un market-maker par aller-retour.
- **Mesure de liquidité** : plus c'est petit, plus c'est liquide.
- **Marchés très liquides** : spread = 1 tick.

### 3.4 Insertion / matching d'ordres

**Insertion de LO** : va au bout de la file de temps (après tous les LOs existants au même prix).

**Matching d'un MO** :
1. Arrivée MO.
2. Match avec les LOs de plus haute priorité prix/temps.
3. Consomme la liquidité aux premiers niveaux (jusqu'à épuisement du MO).
4. LOB résultant : niveaux consommés ou amincis.

### 3.5 LO vs MO — arbitrage

**LO (posting)** :
- Meilleur prix (gagne le spread).
- Exécution **retardée** et **incertaine**.
- ⇒ **Risque de marché** (le prix peut bouger avant exécution) + **risque d'exécution**.

**MO (taking)** :
- Pire prix (paye le spread).
- Exécution **immédiate** et **certaine**.
- ⇒ **Pas de risque de marché** ni d'exécution.

**À l'examen** : chaque acteur choisit selon ses objectifs. Un market-maker préfère les LOs (gagner le spread). Un trader informé a besoin de MOs (rapide, certain). C'est une tension fondamentale.

### 3.6 Mécanisme de double enchère

**Objectif** : simultanément (i) former un prix d'équilibre, (ii) disséminer l'information sur ce prix.

**Une journée typique** :
- Commence par un **fixing d'ouverture**.
- Phase de **continuous auction** (enchères continues) pendant la journée.
- **Mid-day auction** sur certains marchés européens et asiatiques.
- **Fixing de clôture**.

**Phase de pré-fixing (~5 min)** : acheteurs et vendeurs insèrent des ordres (quantité à un prix donné). Ex : bid order de 100 à 10 = « j'accepte d'acheter 100 au prix ≤ 10 ». En fin de phase, le prix est fixé à un **équilibre walrassien**.

### 3.7 Équilibre walrassien

Soit $f_{LOB}^-(s)$ le volume côté vendeurs au prix $s$ et $f_{LOB}^+(s)$ côté acheteurs. Fonctions cumulées :
$$F_{LOB}^+(S) = \int_{-\infty}^S f_{LOB}^+(s)\, ds, \qquad F_{LOB}^-(S) = \int_S^{+\infty} f_{LOB}^-(s)\, ds$$

Prix fixé à $S^*$ tel que :
$$\boxed{F_{LOB}^+(S^*) = F_{LOB}^-(S^*)}$$

**Interprétation** : on cherche le prix qui **équilibre** la demande cumulée (tous les acheteurs prêts à payer ≥ $S^*$) et l'offre cumulée (tous les vendeurs prêts à céder à ≤ $S^*$). C'est le prix qui **maximise le volume tradé**.

**À noter** : un fixing concentre l'information et fige un prix unique pour une grosse masse d'ordres. C'est un mécanisme différent de l'auction continue où chaque MO déclenche une transaction à un prix potentiellement différent.

## Chapitre 4 — Indicateurs intraday (BLOC CENTRAL DEMANDÉ)

### 4.1 Dynamique du prix intraday

**À l'échelle journée** (ex. Alstom, 1er février 2011) : le prix tradé semble **continu** et a une allure **mean-reverting**.

**À l'échelle 25 minutes** : on voit la **discrétisation** — le prix évolue sur une grille. L'incrément minimal est le **tick**, dont la taille dépend du niveau de prix et (depuis MiFID II) de l'activité de trading.

### 4.2 Intraday patterns — importance et causes

**Importance** :
1. Critiques pour l'exécution vs un benchmark.
2. La **concentration du trading** dans certaines plages horaires reflète à quel point les prix sont **informatifs** à ces moments.
3. Il y a une **incitation** pour les traders informés ET pour les traders de liquidité à timer leurs trades aux mêmes moments.

**Causes des patterns** :
1. **Ouverture d'un marché lié** (ex. ouverture US influe sur marchés européens).
2. **Diffusion de news à heure fixe** (macro, earnings — souvent avant l'ouverture).
3. **Calcul des prix de référence** : selon qu'il y a fixing, comment il est calculé, etc.

### 4.3 Courbes de volume intraday

**Forme en U caractéristique** : plus de volume à l'ouverture et à la clôture qu'au milieu.

**Asymétrie** : typiquement **plus de volume à la clôture qu'à l'ouverture**, en raison de :
- Ajustement d'inventaire en fin de journée.
- Stratégies intraday qui se débouclent avant clôture.

**US (NYSE)** : forme en U classique. Pic fort à la clôture.

**Europe (Euronext-Paris, LSE)** : forme en U avec :
- **Spikes** en milieu d'après-midi dus à l'**ouverture US** (14h30 GMT pour LSE, 15h30 pour Paris).
- Effet des **news macro US** (souvent 13h30 ou 14h30 GMT).
- Effet des **news ECB**.
- Expirations de dérivés (troisième vendredi du mois, etc.).

**Asie (HKE, TSE)** :
- **Pause déjeuner** clairement visible au milieu.
- **TSE** : changements de politique en 2006 (début après-midi à 13h puis retour à 12h30 en avril).
- **HKE** : forme en **J** (plus qu'en U) car 17% du volume journalier se fait dans les 15 dernières minutes, dont un quart dans la dernière minute (calcul médiane sur 5 prix à 15s d'intervalle pour le prix de clôture).

### 4.4 Impact de l'ouverture d'un marché lié

**Trois mécanismes** :
1. **Substitution** : investisseurs tradant sur le marché nouvellement ouvert décident de trader l'autre aussi → volume supplémentaire.
2. **Information** : les prix d'ouverture apportent une nouvelle info qui doit être intégrée dans les prix de l'autre marché.
3. **Arbitrage** : différences de prix temporaires entre les deux marchés → volume supplémentaire pour les arbitrer.

**Ouverture US (9h30 US eastern)** :
- = 15h30 pour Euronext-Paris.
- = 14h30 pour LSE.

### 4.5 News macro

**BLS (Bureau of Labor Statistics)** : rapport mensuel sur l'emploi US, 1er vendredi du mois à 8h30 US eastern. Indicateur phare : **non-farm payroll employment** (nombre d'emplois créés/détruits hors secteur agricole). Énorme pic de volume juste après la release.

**ISM manufacturing** : 10h US eastern le 1er jour ouvré du mois. Composite index indiquant croissance/déclin manufacturier.

**FOMC (Fed)** : 8 fois par an, décisions de politique monétaire. Énorme volatilité autour de l'annonce.

### 4.6 Courbe intraday du nombre de trades

**Forme proche d'une U**, similaire au volume. Distribution des trades concentrée à l'ouverture et à la clôture.

### 4.7 Courbe intraday de volatilité

**Forme en U également**, mais **asymétrique dans l'autre sens** : la volatilité est **plus forte à l'ouverture qu'à la clôture** (contrairement au volume).

**Pourquoi** ?
- **Ouverture** : digestion des news de la nuit, des marchés asiatiques, des events weekend/nuit. Beaucoup d'incertitude → volatilité élevée.
- **Milieu de journée** : marché calme, peu de news, faible vol.
- **Clôture** : regain d'activité (volume), mais moins d'incertitude informationnelle que le matin → vol moins forte.

**À l'examen** : bien opposer volume (asymétrie vers la clôture) et volatilité (asymétrie vers l'ouverture).

### 4.8 Courbe intraday du spread bid-ask

**Forme en S** caractéristique :
- **Spread élevé à l'ouverture** (grande incertitude, market-makers se protègent).
- **Déclin** au fur et à mesure que le marché se stabilise.
- **Plat** au milieu de journée.
- **Décline** à nouveau en fin de journée.

**Baisse de fin de journée** : due à la **hausse du volume** à la clôture → les market-makers veulent **liquider passivement** leur inventaire, donc ils resserrent le spread pour attirer la contrepartie.

**Effet MiFID / Reg NMS / HFT** : **ré-étroitissement** du spread au début et à la fin de journée (plus d'activité), et léger **élargissement** en milieu de journée. C'est l'empreinte de l'activité HFT.

### 4.9 Quantité au best quote

**Forme en J** : faible au début, augmente au cours de la journée, pic à la fin. Reflète l'accumulation de LOs dans le carnet.

### 4.10 Synthèse : les 4 courbes intraday à savoir décrire

| Indicateur | Forme | Asymétrie | Cause principale |
|------------|-------|-----------|------------------|
| **Volume** | U | Plus élevé à la clôture | Ajustement d'inventaire, stratégies intraday |
| **Nombre de trades** | U | Similaire au volume | Cohérent avec volume |
| **Volatilité** | U | Plus élevée à l'ouverture | Digestion de l'info overnight |
| **Spread** | S (U renversé possible avec HFT) | Moins de spread fin de journée | Liquidation passive market-makers |
| **Quantité au best** | J | Plus à la clôture | Accumulation de LOs |

**La relation MRR** ($\varphi \propto \sigma / \sqrt N$) permet de comprendre **pourquoi** ces courbes co-évoluent :
- À l'ouverture : **haute vol** + **faible $N$** → spread très large.
- Milieu de journée : vol moyenne + $N$ moyen → spread moyen.
- Clôture : vol moyenne + **haut $N$** → spread faible.

---

# SYNTHÈSE TRANSVERSALE POUR L'EXAMEN

## Les 10 formules à connaître par cœur

1. **Intensité d'un processus ponctuel** : $\lambda(t|\mathcal{F}_t) = \lim_{h\downarrow 0}\mathbb E[(N_{t+h}-N_t)/h | \mathcal{F}_t]$.
2. **Hawkes 1D exponentiel** : $\lambda(t) = \lambda_0 + \alpha \sum_{\tau_i < t} e^{-\beta(t - \tau_i)}$.
3. **Stationnarité 1D** : $\alpha/\beta < 1$. Intensité moyenne : $\mathbb E[\lambda] = \lambda_0/(1 - \alpha/\beta)$.
4. **Stationnarité multi-D** : rayon spectral $\rho(\boldsymbol\Gamma) < 1$, avec $\Gamma_{mn} = \alpha^{mn}/\beta^{mn}$.
5. **Log-vraisemblance processus ponctuel** : $\ln L = \int_0^T(1-\lambda(s))ds + \int_0^T \ln \lambda(s) dN_s$.
6. **Robbins-Monro** : $\theta_{n+1} = \theta_n - \gamma_{n+1}(h(\theta_n) + \Delta M_{n+1})$, $\sum \gamma_n = \infty$, $\sum \gamma_n^2 < \infty$.
7. **Ruppert-Polyak** : $\bar\theta_n = (1/n)\sum_{k<n}\theta_k$, avec $\gamma_n = c/n^\vartheta$, $\vartheta \in (1/2, 1)$.
8. **Variance réalisée** : $\widehat V_R(\Delta) = \sum_j (\ln S_{\Delta j} - \ln S_{\Delta(j-1)})^2 \to \int \sigma^2 ds$.
9. **Hayashi-Yoshida** : $U_n = \sum_{i,j} \Delta P^1(I_i^1) \Delta P^2(I_j^2) \mathbf{1}_{I_i^1 \cap I_j^2 \neq \emptyset}$.
10. **MRR** : $\varphi = c \sigma_{\text{mid}} / \sqrt N$.

## Les 5 questions-type d'examen avec plan de réponse

### Q1. « Expliquez le processus de Hawkes 1D et ses propriétés. »

**Plan** :
1. **Motivation** : clustering empirique incompatible avec Poisson homogène.
2. **Définition** : intensité $\lambda(t) = \lambda_0 + \int \varphi(t-s) dN_s$, noyau positif décroissant.
3. **Noyau exponentiel** : $\varphi(t) = \alpha e^{-\beta t}$. Interprétation de $\alpha$ (saut d'excitation) et $\beta$ (mémoire).
4. **Branching ratio** : $\alpha/\beta$ = # d'événements déclenchés en moyenne par un événement. Stationnarité ⇔ $\alpha/\beta < 1$. Marché endogène si proche de 1.
5. **Intensité moyenne stationnaire** : $\lambda_0/(1 - \alpha/\beta)$.
6. **Simulation** (Ogata 1981) : thinning / acceptation-rejet.
7. **Estimation** : MLE par log-vraisemblance récursive (Ogata). Consistance, normalité asymptotique, efficacité.
8. **Test de simulation** : QQ-plot des $\Lambda(\tau_{i-1}, \tau_i)$ contre $\mathcal{E}(1)$.

### Q2. « Expliquez le Hawkes 2D et comment il modélise la mean-reversion haute fréquence. »

**Plan** :
1. **Généralisation** : matrice de noyaux $\Phi_{mn}$ avec auto/cross-excitation.
2. **Stationnarité** : $\rho(\Gamma) < 1$ avec $\Gamma_{mn} = \alpha^{mn}/\beta^{mn}$.
3. **Modèle de Bacry** : $p(t) = N^1(t) - N^2(t)$, avec uniquement cross-excitation et symétrie.
4. **Interprétation** : un saut à la hausse augmente la probabilité d'un saut à la baisse futur (et inversement). Ceci **crée la mean-reversion**.
5. **Signature plot** : $C(\tau) = \Lambda[\kappa^2 + (1-\kappa^2)(1-e^{-\gamma\tau})/(\gamma\tau)]$. Quand $\tau \to 0$, $C(\tau) \to \Lambda$ ; quand $\tau \to \infty$, $C(\tau) \to \Lambda\kappa^2 < \Lambda$. → **signature plot décroissant**, comme empirique.
6. **Effet Epps** : version 4D donne $\rho(\tau) \to 0$ quand $\tau \to 0$ (explique la corrélation qui s'effondre en HF).

### Q3. « Expliquez les estimateurs de volatilité HF et le problème du bruit. »

**Plan** :
1. **Estimateur naïf** : $\widehat V_R(\Delta) = \sum (\Delta \ln S)^2 \to \int \sigma^2 ds$ quand $\Delta \to 0$ (dans le modèle idéal).
2. **Problème empirique** : le signature plot **diverge** en $\Delta \to 0$.
3. **Raison** : $X = M + \varepsilon$ où $\varepsilon$ = bruit de microstructure (bid-ask bounce, tick, asynchronisme).
4. **Décomposition** : $\widehat V_R = \sum (r_\Delta^M)^2 + \sum \eta_\Delta^2 + 2\sum r^M \eta$. Le terme $\sum \eta^2 \sim T/\Delta \to \infty$.
5. **Alternatives** : Garman-Klass (OHLC), multi-scale (Aït-Sahalia-Jacod), TSRV (Zhang), pré-averaging (Jacod).
6. **Formule GK** : $\sigma_{GK}^2 = (1/2)(\max - \min)^2 - (2\ln 2 - 1)(\text{close} - \text{open})^2$.

### Q4. « Expliquez l'estimateur de Hayashi-Yoshida. »

**Plan** :
1. **Problème** : l'estimateur classique de covariance $\sum r^1 r^2$ requiert des données **synchrones**.
2. **Effet Epps** : si on synchronise et on fait $\Delta \to 0$, la corrélation mesurée tend vers 0 même si la corrélation sous-jacente est $\rho > 0$.
3. **Idée HY** : utiliser les intervalles naturels $I_i^1, I_j^2$ et sommer les produits d'increments uniquement quand les intervalles **se chevauchent** : $U_n = \sum_{i,j} \Delta P^1(I_i^1) \Delta P^2(I_j^2) \mathbf{1}_{I_i^1 \cap I_j^2 \neq \emptyset}$.
4. **Corrélation** : deux formes, $R^1 = U_n/(T\sigma_1\sigma_2)$ (si $\sigma$ connues) ou $R^2 = U_n / (\sqrt{\sum (\Delta P^1)^2}\sqrt{\sum (\Delta P^2)^2})$.
5. **Propriétés** : convergent vers $\rho$, pas de choix de $\Delta$. **Mais** : pas robuste au bruit de microstructure.

### Q5. « Décrivez les courbes intraday (volume, volatilité, spread). »

**Plan** :
1. **Volume** : **U-shape**, asymétrie vers la clôture (ajustement d'inventaire, stratégies intraday). Spikes dus à l'ouverture US (Europe), news macro, expirations.
2. **Volatilité** : **U-shape** aussi, mais asymétrie inverse — **plus forte à l'ouverture** (digestion info overnight). Milieu de journée plat.
3. **Spread** : **S-shape** classique (haut à l'ouverture, déclin, plat, re-déclin). HFT et électronisation ont créé un creusement supplémentaire en début et fin de journée.
4. **Relation MRR** : $\varphi \propto \sigma / \sqrt{N}$ explique la cohérence : haute vol + faible $N$ à l'ouverture → spread large ; clôture : vol modérée + fort $N$ → spread étroit.
5. **Quantité au best** : **J-shape** (accumulation LOs au cours de la journée).
6. **Large tick** : la relation MRR échoue (spread bloqué à 1 tick). Dayri-Rosenbaum proposent des modèles avec zones d'incertitude.

---

# PLAN DE RÉVISION SUR 10-12 HEURES

## Jour 1 (4-5h) — HFT4 Hawkes
- Heure 1 : processus ponctuels, Poisson, intensité, temps changement.
- Heure 2 : Hawkes 1D (définition, stationnarité, noyau exp).
- Heure 3 : simulation Ogata + test QQ-plot + log-vraisemblance récursive.
- Heure 4 : Hawkes multidim (stationnarité spectrale, simulation).
- Heure 5 : applications microstructure (modèle Bacry 1D/2D, signature plot, Epps, QHawkes).

## Jour 2 (3-4h) — HFT3 Approximation stochastique
- Heure 1 : cadre général Robbins-Monro + conditions sur le pas.
- Heure 2 : convergence (martingale, ODE) + vitesses de convergence.
- Heure 3 : Ruppert-Polyak + application dark pools.
- Heure 4 : application posting d'ordres limite (expression C', conditions de convexité, algo).

## Jour 3 (3h) — HFT2 + HFT1
- Heure 1 : HFT2 — variance réalisée, signature plot, Garman-Klass, multi-scale.
- Heure 2 : HFT2 — covariance classique, effet Epps, Hayashi-Yoshida, algorithmes de trading (TWAP, PoV, VWAP, IS).
- Heure 3 : HFT1 — LOB, mécanismes, fixing, courbes intraday (volume, vol, spread), MRR.

## Dernières heures (1-2h)
- Relire la synthèse transversale (10 formules + 5 questions-type).
- Refaire de tête les preuves courtes : stationnarité Hawkes 1D, log-vraisemblance processus ponctuel, équilibre walrassien, dérivation MRR intuitive.

---

**Bonne révision ! Tu as tout ce qu'il te faut pour l'examen. N'hésite pas à revenir sur les sections les plus techniques (Hawkes 1D stationnarité, récursivité log-vraisemblance, effet signature plot, Hayashi-Yoshida) : ce sont les points où le correcteur évaluera le mieux la compréhension profonde.**
