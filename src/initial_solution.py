import numpy as np

from src.results import save_results
from src.dd_generator import generate_due_dates_brah, generate_weights
from src.plots import plot_gantt
from src.scheduler import compute_objectives
from config import FOLDE_MAP


def taillard_sequences(datasets):
    """
    Exécute la séquence identitaire pour toutes les instances
    et sauvegarde les Gantt dans data/gantt/tai_gantt/
    """

    print(f"\n{'='*50}")
    print("Séquences Taillard...")

    for name, instances in datasets.items():
        folder = FOLDE_MAP.get(name, name)
        for idx, inst in enumerate(instances):
            pt        = inst['processing_times']
            due_dates = generate_due_dates_brah(inst, tau=2)
            weights   = generate_weights(inst)
            n_jobs    = inst['n_jobs']
            sequence  = list(range(n_jobs))
            instance_id = f"instance_{idx+1}"

            obj = compute_objectives(sequence, pt, due_dates, weights)

            print(f"\n  {name} — Instance {idx+1}:")
            """print(f"    Séquence : {[j+1 for j in sequence]}")
            print(f"    TT       : {obj['TT']}")
            print(f"    TWT      : {obj['TWT']}")
            print(f"    T_max    : {obj['T_max']}")
            print(f"    NT       : {obj['NT']}")"""

            save_results(
                sequence         = sequence,
                processing_times = pt,
                due_dates        = due_dates,
                weights          = weights,
                filepath         = f"resultats/taillard/{folder}/{instance_id}.csv"
            )

            """plot_gantt(
                sequence         = sequence,
                processing_times = pt,
                due_dates        = due_dates,
                weights          = weights,
                title            = f"Taillard — {name} Instance {idx+1}",
                filename         = f"gantts/taillard/{folder}/instance_{idx+1}.png"
            )"""

"""
1. Trier les jobs par due date croissante → ordre EDD
2. Prendre les 2 premiers jobs → tester les 2 permutations
   → garder la meilleure selon TT (ou TWT, T_max, NT)
3. Pour chaque job suivant :
   → tester toutes les positions d'insertion
   → garder la meilleure séquence
4. Retourner la séquence finale
"""

def nehedd(processing_times, due_dates, weights=None, objective='TT'):
    """
    Heuristique NEHedd pour le PFSP avec due dates.
    Solution initiale pour la métaheuristique IG-TS.

    Args:
        processing_times : matrice (n_machines x n_jobs)
        due_dates        : array (n_jobs,)
        weights          : array (n_jobs,) — optionnel pour TWT
        objective        : 'TT', 'TWT', 'T_max', 'NT'

    Returns:
        sequence : liste des jobs ordonnés
    """
    n_jobs = processing_times.shape[1]

    # Étape 1 — Trier par due date croissante (EDD)
    jobs_sorted = sorted(range(n_jobs), key=lambda j: due_dates[j])

    # Étape 2 — Initialiser avec le premier job
    sequence = [jobs_sorted[0]]

    # Étape 3 — Insérer chaque job à la meilleure position
    for k in range(1, n_jobs):
        job     = jobs_sorted[k]
        best_seq   = None
        best_value = float('inf')

        # Tester toutes les positions d'insertion
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            obj = compute_objectives(
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


def nehEdd(datasets):
    """
    Résout toutes les instances avec NEHedd pour chaque objectif
    et sauvegarde les Gantt dans data/gantt/nehedd/
    """
    print(f"\n{'='*50}")
    print("Séquences NEH_EDD...")

    for name, instances in datasets.items():
        folder = FOLDE_MAP.get(name, name)
        for idx, inst in enumerate(instances):
            pt        = inst['processing_times']
            due_dates = generate_due_dates_brah(inst, tau=2)
            weights   = generate_weights(inst)
            instance_id = f"instance_{idx+1}"

            print(f"\n  {name} — Instance {idx+1}:")

            for objective in ['TT', 'TWT', 'T_max', 'NT']:
                sequence = nehedd(pt, due_dates, weights, objective=objective)
                obj      = compute_objectives(sequence, pt, due_dates, weights)

                """print(f"    [{objective}] Séquence : {[j+1 for j in sequence]}")
                print(f"           TT={obj['TT']}, TWT={obj['TWT']}, "
                      f"T_max={obj['T_max']}, NT={obj['NT']}")"""
                
                save_results(
                    sequence         = sequence,
                    processing_times = pt,
                    due_dates        = due_dates,
                    weights          = weights,
                    filepath = f"resultats/nehedd/{folder}/{instance_id}_{objective}.csv"
                )

                """plot_gantt(
                    sequence         = sequence,
                    processing_times = pt,
                    due_dates        = due_dates,
                    weights          = weights,
                    title            = f"NEHedd [{objective}] — {name} Instance {idx+1}",
                    filename         = f"gantts/nehedd/{folder}/instance_{idx+1}_{objective}.png"
                )"""