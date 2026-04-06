import numpy as np


def generate_due_dates(instance, T, R, seed=None):
    """
    Génère les due dates selon Potts & Van Wassenhove.
    
    Args:
        instance : dict avec 'lb', 'n_jobs', 'processing_times'
        T : tardiness factor (ex: 0.2, 0.4, 0.6, 0.8)
        R : due date range (ex: 0.2, 0.6, 1.0)
        seed : pour reproductibilit
    
    Returns:
        due_dates : array de taille n_jobs
    """
    if seed is not None:
        np.random.seed(seed)
    
    P = instance['lb']  # borne inférieure du makespan
    n = instance['n_jobs']
    
    lower = P * (1 - T - R/2)
    upper = P * (1 - T + R/2)
    
    # S'assurer que les bornes sont positives
    lower = max(1, lower)
    upper = max(lower + 1, upper)
    
    due_dates = np.random.uniform(lower, upper, size=n)
    due_dates = np.round(due_dates).astype(int)
    
    return due_dates


def generate_all_scenarios(instance, seed=42):
    """
    Génère tous les scénarios (T, R) pour une instance.
    Retourne un dict avec les due dates pour chaque scénario.
    """
    T_values = [0.2, 0.4, 0.6, 0.8]
    R_values = [0.2, 0.6, 1.0]
    
    scenarios = {}
    for T in T_values:
        for R in R_values:
            key = f"T{T}_R{R}"
            scenarios[key] = generate_due_dates(instance, T, R, seed=seed)
    return scenarios


def generate_due_dates_brah(instance, tau=2):
    pt = instance['processing_times']  # (n_machines, n_jobs)
    
    # Somme des temps de traitement par job
    total_work = pt.sum(axis=0)  # shape: (n_jobs,)
    
    due_dates = tau * total_work
    
    return due_dates.astype(int)


def generate_weights(instance, seed=42):
    """
    Génère les poids w_j pour chaque job.
    w_j ~ U[1, 10] (distribution uniforme entière)

    Args:
        instance : dict avec 'n_jobs'
        seed     : pour reproductibilité

    Returns:
        weights : array de taille n_jobs
    """
    np.random.seed(seed)
    n_jobs  = instance['n_jobs']
    weights = np.random.randint(1, 11, size=n_jobs)
    return weights