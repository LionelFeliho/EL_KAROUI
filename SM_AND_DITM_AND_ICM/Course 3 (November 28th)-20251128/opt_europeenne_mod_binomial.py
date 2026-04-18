def recursion(r, u, d, fonction_prix_t_i_plus_one, delta_flag=0):
    """
    On effectue une itération de la date t_{i+1} à la date t_i.
    
    Paramètres
    + r, u, d: les paramètres du modèle binomial 
    + fonction_prix_t_i_plus_one (une fonction Python):
        La fonction de prix v(t_{i+1}, .) à la date t_{i+1}.
        Doit prendre comme argument la valeur S_{t_{i+1}} du sous-jacent à la date t_{i+1}.
    
    Output:
    + Une fonction Python, la fonction de prix v(t_i, .) à la date t_i.
      Doit prendre comme argument la valeur S_{t_i} du sous-jacent.
            
    + Si delta_flag: renvoie une autre fonction Python, le delta delta(t_i, .) du portefeuille à la date t_i.
      Doit prendre comme argument la valeur S_{t_i} du sous-jacent.
    """
    q_up = (1 + r - d) / (u - d)
    q_down = 1 - q_up
    
    def fonction_prix_t_i(S):
        valeur = (fonction_prix_t_i_plus_one(S*u)*q_up + fonction_prix_t_i_plus_one(S*d)*q_down) / (1+r)
        return valeur
    
    if delta_flag:
            def delta_t_i(S):
                delta = (fonction_prix_t_i_plus_one(S*u) - fonction_prix_t_i_plus_one(S*d)) / (S*(u - d))
                return delta
   
    if delta_flag == 0:
        return fonction_prix_t_i
    
    else:
        return fonction_prix_t_i, delta_t_i


def recursion_n_i(r, u, d, payoff, i, n, delta_flag=0):
    """
    On effectue les itérations de la date t_n à à la date t_i, i < n,
    pour l'option de maturité t_n et de payoff = payoff(S_{t_n})
    
    Output: une fonction Python, la fonction de prix v(t_i, .) à la date t_i.
            Cette fonction prendra comme argument la valeur S_{t_i} du sous-jacent.
            
    Output:
    + Une fonction Python, la fonction de prix v(t_i, .) à la date t_i.
            
    + Si delta_flag: on renvoie la fonction de prix v(t_i, .) ET le delta(t_i, .)
      du portefeuille de couverture à la date t_i.
      Ces fonctions prendront comme argument la valeur S_{t_i} du sous-jacent.
    
    """
    fonction_prix_et_delta = [payoff, 0]
        
    for j in range(n, i, -1):
        fonction_prix_et_delta = recursion(r, u, d, fonction_prix_et_delta[0], delta_flag=1)
        ## la variable fonction_prix_et_delta contient un couple de fonctions
    
    if delta_flag == 0:
        return fonction_prix_et_delta[0]
    
    else:
        return fonction_prix_et_delta
    
    