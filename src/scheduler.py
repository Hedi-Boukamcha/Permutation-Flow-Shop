import numpy as np


def compute_completion_times(sequence, processing_times):
    """
    Calcule les completion times pour une séquence donnée.
    
    Args:
        sequence        : liste des jobs dans l'ordre ex: [3, 1, 4, 2, ...]
        processing_times: matrice (n_machines x n_jobs)
    
    Returns:
        C : matrice (n_machines x n_jobs) des completion times
    """
    n_machines, _ = processing_times.shape

    C = np.zeros((n_machines, len(sequence)), dtype=int)

    for j, job_idx in enumerate(sequence):
        for i in range(n_machines):
            if i == 0 and j == 0:
                C[i][j] = processing_times[i][job_idx]
            elif i == 0:
                C[i][j] = C[i][j-1] + processing_times[i][job_idx]
            elif j == 0:
                C[i][j] = C[i-1][j] + processing_times[i][job_idx]
            else:
                C[i][j] = max(C[i-1][j], C[i][j-1]) + processing_times[i][job_idx]

    return C


def compute_objectives(sequence, processing_times, due_dates, weights=None):
    """
    Calcule tous les objectifs due-date pour une séquence donnée.

    Args:
        sequence         : liste des jobs dans l'ordre
        processing_times : matrice (n_machines x n_jobs)
        due_dates        : array (n_jobs,) des due dates
        weights          : array (n_jobs,) des poids w_j
                           Si None → poids = 1 pour tous les jobs

    Returns:
        dict avec TT, TWT, T_max, NT, Cj, Tj, Uj
    """
    n_jobs = len(sequence)

    # Poids par défaut = 1
    if weights is None:
        weights = np.ones(n_jobs, dtype=int)

    # Calcul des completion times
    C = compute_completion_times(sequence, processing_times)

    # Completion time de chaque job = dernière machine
    Cj = C[-1]  # shape: (n_jobs,)

    # Due dates et poids dans l'ordre de la séquence
    dj = due_dates[list(sequence)]
    wj = 1 #weights[list(sequence)]

    # Tardiness : T_j = max{0, C_j - d_j}
    Tj = np.maximum(0, Cj - dj)

    # Tardiness indicator : U_j = 1 si T_j > 0, sinon 0
    Uj = (Tj > 0).astype(int)

    # Objectifs
    TT    = int(np.sum(Tj))           # Total Tardiness       = Σ T_j
    TWT   = int(np.sum(wj * Tj))      # Total Weighted Tard.  = Σ w_j × T_j
    T_max = int(np.max(Tj))           # Maximum Tardiness     = max T_j
    NT    = int(np.sum(Uj))           # Number of Tardy Jobs  = Σ U_j

    return {
        "TT":    TT,
        "TWT":   TWT,
        "T_max": T_max,
        "NT":    NT,
        "Cj":    Cj,   # completion times
        "Tj":    Tj,   # tardiness par job
        "Uj":    Uj    # tardiness indicator par job
    }


