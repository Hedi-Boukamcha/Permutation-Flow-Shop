import os
import numpy as np
from src.dd_generator import generate_due_dates_brah, generate_weights
from src.initial_solution import nehEdd, nehedd
from src.scheduler import compute_objectives
from src.results import save_results
from src.plots import plot_gantt


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


# ─────────────────────────────────────────────────────────
# ITERATED GREEDY for one instance
# ─────────────────────────────────────────────────────────

def ig(processing_times, due_dates, weights=None, objective='TT',
       k=4, max_iter=100, filepath=None):
    """
    Iterated Greedy pour le PFSP avec due dates.

    Args:
        processing_times : matrice (n_machines x n_jobs)
        due_dates        : array (n_jobs,)
        weights          : array (n_jobs,)
        objective        : 'TT', 'TWT', 'T_max', 'NT'
        k                : nombre de jobs à retirer (destruction)
        max_iter         : nombre d'itérations maximum

    Returns:
        best_sequence : meilleure séquence trouvée
        best_value    : valeur de l'objectif
        history       : historique des valeurs par itération
    """
    # 1. Solution initiale NEHedd
    sequence = nehedd(processing_times, due_dates, weights, objective)
    obj      = compute_objectives(sequence, processing_times, due_dates, weights)

    best_sequence = sequence[:]
    best_value    = obj[objective]
    current_value = best_value
    history       = [best_value]

    print(f"  IG démarrage — {objective} initial : {best_value}")

    # 2. Boucle principale
    for iteration in range(max_iter):

        # a. Destruction — retirer les k jobs les plus tardifs
        partial_seq, removed_jobs = destruction(
            sequence         = sequence,
            processing_times = processing_times,
            due_dates        = due_dates,
            weights          = weights,
            k                = k,
            objective        = objective
        )

        # b. Reconstruction — réinsérer selon meilleure position
        new_sequence = reconstruction(
            partial_sequence = partial_seq,
            removed_jobs     = removed_jobs,
            processing_times = processing_times,
            due_dates        = due_dates,
            weights          = weights,
            objective        = objective
        )

        # c. Évaluer la nouvelle séquence
        new_obj   = compute_objectives(
            new_sequence, processing_times, due_dates, weights
        )
        new_value = new_obj[objective]

        # d. Acceptation — garder si meilleur ou égal
        if new_value <= current_value:
            sequence      = new_sequence[:]
            current_value = new_value

        # e. Mettre à jour la meilleure solution globale
        if new_value < best_value:
            best_sequence = new_sequence[:]
            best_value    = new_value

        history.append(best_value)

        # Afficher progression toutes les 10 itérations
        if (iteration + 1) % 10 == 0:
            print(f"  Itération {iteration+1:4d} — {objective} : {best_value}")
        
        # 3. Sauvegarde résultats CSV
    if filepath:
        save_results(
            sequence         = best_sequence,
            processing_times = processing_times,
            due_dates        = due_dates,
            weights          = weights,
            filepath         = filepath
        )

    # 4. Sauvegarde Gantt
    """if filepath:
        gantt_path = filepath.replace("resultats", "gantts").replace(".csv", ".png")
        plot_gantt(
            sequence         = best_sequence,
            processing_times = processing_times,
            due_dates        = due_dates,
            weights          = weights,
            title            = f"IG [{objective}] — {os.path.basename(filepath)}",
            filename         = gantt_path
        )"""

    print(f"  IG terminé — {objective} final : {best_value}")

    return best_sequence, best_value, history

# ─────────────────────────────────────────────────────────
# ITERATED GREEDY for all instances
# ─────────────────────────────────────────────────────────

def runIG(datasets):
    """
    ORCHESTRATION — résout TOUTES les instances avec IG
    Génère les Gantt + CSV
    """
    folder_map = {
        "tai20j_5m":  "20j_5m",
        "tai50j_10m": "50j_10m"
    }

    for name, instances in datasets.items():
        folder = folder_map.get(name, name)

        for idx, inst in enumerate(instances):
            pt        = inst['processing_times']
            due_dates = generate_due_dates_brah(inst, tau=2)
            weights   = generate_weights(inst)

            print(f"\n  {name} — Instance {idx+1}:")

            for objective in ['TT', 'TWT', 'T_max', 'NT']:
                best_seq, best_val, history = ig(
                    processing_times = pt,
                    due_dates        = due_dates,
                    weights          = weights,
                    objective        = objective,
                    k                = 4,
                    max_iter         = 100
                )

                obj = compute_objectives(best_seq, pt, due_dates, weights)
                print(f"    [{objective}] TT={obj['TT']}, TWT={obj['TWT']}, "
                      f"T_max={obj['T_max']}, NT={obj['NT']}")

                # Résultats CSV
                save_results(
                    sequence         = best_seq,
                    processing_times = pt,
                    due_dates        = due_dates,
                    weights          = weights,
                    filepath = f"resultats/ig/{folder}/instance_{idx+1}_{objective}.csv"
                )

                # Gantt
                """plot_gantt(
                    sequence         = best_seq,
                    processing_times = pt,
                    due_dates        = due_dates,
                    weights          = weights,
                    title  = f"IG [{objective}] — {name} Instance {idx+1}",
                    filename = f"gantts/ig/{folder}/instance_{idx+1}_{objective}.png"
                )"""

