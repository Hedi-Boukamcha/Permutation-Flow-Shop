import numpy as np
from src.initial_solution import nehEdd
from src.scheduler import compute_objectives


"""1. Solution initiale  → NEHedd
2. Répéter jusqu'au critère d'arrêt :
   a. DESTRUCTION  → retirer k jobs les plus tardifs
   b. RECONSTRUCTION → réinsérer à la meilleure position
   c. ACCEPTATION  → garder si meilleur
3. Retourner meilleure solution"""

# ─────────────────────────────────────────────────────────
# DESTRUCTION
# ─────────────────────────────────────────────────────────

def destruction(sequence, processing_times, due_dates, weights, k, objective):
    """
    Retire les k jobs les plus tardifs de la séquence.

    Args:
        sequence         : liste des jobs dans l'ordre
        processing_times : matrice (n_machines x n_jobs)
        due_dates        : array (n_jobs,)
        weights          : array (n_jobs,)
        k                : nombre de jobs à retirer
        objective        : 'TT', 'TWT', 'T_max', 'NT'

    Returns:
        partial_sequence : séquence sans les k jobs retirés
        removed_jobs     : liste des k jobs retirés
    """
    # Calculer la tardiness de chaque job dans la séquence
    obj = compute_objectives(sequence, processing_times, due_dates, weights)
    Tj  = obj['Tj']

    # Trier les positions par tardiness décroissante
    # → les jobs les plus tardifs en premier
    positions_by_tardiness = sorted(
        range(len(sequence)),
        key     = lambda pos: Tj[pos],
        reverse = True
    )

    # Positions des k jobs les plus tardifs
    positions_to_remove = set(positions_by_tardiness[:k])

    # Construire séquence partielle et jobs retirés
    partial_sequence = [
        sequence[pos] for pos in range(len(sequence))
        if pos not in positions_to_remove
    ]
    removed_jobs = [
        sequence[pos] for pos in sorted(positions_to_remove)
    ]

    return partial_sequence, removed_jobs


# ─────────────────────────────────────────────────────────
# RECONSTRUCTION
# ─────────────────────────────────────────────────────────

def reconstruction(partial_sequence, removed_jobs, processing_times, due_dates, weights, objective):
    """
    Réinsère les jobs retirés un par un à la meilleure position.

    Args:
        partial_sequence : séquence sans les jobs retirés
        removed_jobs     : jobs à réinsérer
        processing_times : matrice (n_machines x n_jobs)
        due_dates        : array (n_jobs,)
        weights          : array (n_jobs,)
        objective        : 'TT', 'TWT', 'T_max', 'NT'

    Returns:
        sequence : séquence complète reconstruite
    """
    # Trier les jobs retirés par due date croissante (EDD)
    # → insérer les jobs urgents en premier
    removed_jobs_sorted = sorted(
        removed_jobs,
        key = lambda j: due_dates[j]
    )

    sequence = partial_sequence[:]

    for job in removed_jobs_sorted:
        best_seq   = None
        best_value = float('inf')

        # Tester toutes les positions d'insertion
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            obj       = compute_objectives(
                sequence         = candidate,
                processing_times = processing_times,
                due_dates        = due_dates,
                weights          = weights
            )
            value = obj[objective]

            if value < best_value:
                best_value = value
                best_seq   = candidate

        sequence = best_seq

    return sequence
