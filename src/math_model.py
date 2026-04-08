from ortools.sat.python import cp_model
import numpy as np
import csv
import os


def solve_milp_tt(processing_times, due_dates, time_limit=300,
                  filepath=None):
    """
    Résout le PFSP avec OR-Tools (CP-SAT).
    Minimise le Total Tardiness (TT).

    Args:
        processing_times : matrice (n_machines x n_jobs)
        due_dates        : array (n_jobs,)
        time_limit       : temps limite en secondes
        filepath         : chemin du fichier CSV de résultats

    Returns:
        dict avec séquence, TT, status
    """
    n_machines = processing_times.shape[0]
    n_jobs     = processing_times.shape[1]

    L = int(np.sum(processing_times)) + 1

    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # ─────────────────────────────────────────────────────
    # VARIABLES
    # ─────────────────────────────────────────────────────

    S = {}
    for j in range(n_jobs):
        for i in range(n_machines):
            S[j, i] = model.NewIntVar(0, L, f"S_{j}_{i}")

    x = {}
    for j in range(n_jobs):
        for k in range(n_jobs):
            if j != k:
                x[j, k] = model.NewBoolVar(f"x_{j}_{k}")

    C = {}
    for j in range(n_jobs):
        C[j] = model.NewIntVar(0, L, f"C_{j}")

    T = {}
    for j in range(n_jobs):
        T[j] = model.NewIntVar(0, L, f"T_{j}")

    # ─────────────────────────────────────────────────────
    # CONTRAINTES
    # ─────────────────────────────────────────────────────

    # (1) x_jk + x_kj = 1
    for j in range(n_jobs):
        for k in range(n_jobs):
            if j != k:
                model.Add(x[j, k] + x[k, j] == 1)

    # (2) S_ji ≥ S_j,i-1 + p_j,i-1
    for j in range(n_jobs):
        for i in range(1, n_machines):
            model.Add(
                S[j, i] >= S[j, i-1] + int(processing_times[i-1][j])
            )

    # (3) S_ki ≥ S_ji + p_ji - (1 - x_jk) × L
    for j in range(n_jobs):
        for k in range(n_jobs):
            if j != k:
                for i in range(n_machines):
                    pji = int(processing_times[i][j])
                    model.Add(
                        S[k, i] >= S[j, i] + pji - L * (1 - x[j, k])
                    )

    # (4) C_j = S_jm + p_jm
    for j in range(n_jobs):
        pjm = int(processing_times[n_machines-1][j])
        model.Add(C[j] == S[j, n_machines-1] + pjm)

    # (5) T_j ≥ C_j - d_j
    for j in range(n_jobs):
        model.Add(T[j] >= C[j] - int(due_dates[j]))

    # (6) T_j ≥ 0
    for j in range(n_jobs):
        model.Add(T[j] >= 0)

    # ─────────────────────────────────────────────────────
    # FONCTION OBJECTIF
    # ─────────────────────────────────────────────────────

    model.Minimize(sum(T[j] for j in range(n_jobs)))

    # ─────────────────────────────────────────────────────
    # RÉSOLUTION
    # ─────────────────────────────────────────────────────

    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    # ─────────────────────────────────────────────────────
    # EXTRACTION DE LA SOLUTION
    # ─────────────────────────────────────────────────────

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:

        job_starts = [(j, solver.Value(S[j, 0])) for j in range(n_jobs)]
        sequence   = [j for j, _ in sorted(job_starts, key=lambda x: x[1])]
        TT         = sum(solver.Value(T[j]) for j in range(n_jobs))
        status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

        # Détails par job
        job_details = []
        for pos, job in enumerate(sequence):
            cj        = solver.Value(C[job])
            tj        = solver.Value(T[job])
            dd        = int(due_dates[job])
            start_m1  = solver.Value(S[job, 0])
            tardy     = "1" if tj > 0 else "0"

            job_details.append({
                'job':      f"J{job+1}",
                'due_date': dd,
                'start_m1': start_m1,
                'cj':       cj,
                'tj':       tj,
                'tardy':    tardy
            })

        print(f"  Status   : {status_str}")
        print(f"  Séquence : {[j+1 for j in sequence]}")
        print(f"  TT       : {TT}")

        # ─────────────────────────────────────────────────
        # SAUVEGARDE CSV
        # ─────────────────────────────────────────────────

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # En-tête
                writer.writerow([
                    "Job", "Due date", "Start M1",
                    "Completion Cj", "Tardiness", "Tardy"
                ])

                # Détails par job
                for d in job_details:
                    writer.writerow([
                        d['job'],
                        d['due_date'],
                        d['start_m1'],
                        d['cj'],
                        d['tj'],
                        d['tardy']
                    ])

                # Résumé global
                writer.writerow([])
                writer.writerow(["Status", "TT", "", "", "", ""])
                writer.writerow([status_str, TT, "", "", "", ""])

            print(f" Résultats : {filepath}")

        return {
            'status':   status_str,
            'sequence': sequence,
            'TT':       TT
        }

    else:
        print(f"  Pas de solution — status : {solver.StatusName(status)}")
        return None