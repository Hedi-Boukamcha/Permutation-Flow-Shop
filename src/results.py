import csv
import os
import numpy as np
from src.scheduler import compute_completion_times, compute_objectives


def save_results(sequence, processing_times, due_dates, weights=None,
                 filepath="resultats/results.csv"):
    """
    Sauvegarde les résultats détaillés par job dans un fichier CSV.

    Colonnes : Job, Due date, Weight, Start M1, Completion Cj, Tardiness, Tardy
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    n_machines = processing_times.shape[0]
    n_jobs     = len(sequence)

    C = compute_completion_times(sequence, processing_times)

    # Start times
    starts = np.zeros((n_machines, n_jobs), dtype=int)
    for j in range(n_jobs):
        for i in range(n_machines):
            if j == 0 and i == 0:
                starts[i][j] = 0
            elif j == 0:
                starts[i][j] = C[i-1][j]
            elif i == 0:
                starts[i][j] = C[i][j-1]
            else:
                starts[i][j] = max(C[i-1][j], C[i][j-1])

    completion_times = C[n_machines-1]

    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # En-tête
        writer.writerow([
            "Job", "Due date", "Weight",
            "Start M1", "Completion Cj",
            "Tardiness", "Tardy"
        ])

        for j, job in enumerate(sequence):
            cj        = int(completion_times[j])
            dd        = int(due_dates[job])
            w         = int(weights[job]) if weights is not None else 1
            start_m1  = int(starts[0][j])
            tardiness = max(0, cj - dd)
            tardy     = "1" if tardiness > 0 else "0"

            writer.writerow([
                f"J{job+1}",
                dd,
                w,
                start_m1,
                cj,
                tardiness,
                tardy
            ])

    # Afficher les objectifs globaux à la fin
    obj = compute_objectives(sequence, processing_times, due_dates, weights)
    with open(filepath, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])
        writer.writerow(["TT", "TWT", "T_max", "NT", "", "", ""])
        writer.writerow([
            obj['TT'], obj['TWT'], obj['T_max'], obj['NT'],
            "", "", ""
        ])

    print(f"Résultats : {filepath}")