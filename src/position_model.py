from ortools.sat.python import cp_model
import numpy as np
import csv
import os
import time


def solve_milp_tt(processing_times, due_dates, time_limit=300, filepath=None):
    """
    PFSP — Minimise Total Tardiness (TT)
    Formulation par positions (x[j,k]) — adapté de exact_method_ORTools.py
    """
    # processing_times : (n_machines x n_jobs) → on transpose pour [job][machine]
    pt         = processing_times.T  # → (n_jobs x n_machines)
    N          = pt.shape[0]         # n_jobs
    M          = pt.shape[1]         # n_machines
    p          = pt.tolist()
    d          = [int(due_dates[j]) for j in range(N)]
    Cmax_bound = int(np.sum(processing_times)) + 1

    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # ─────────────────────────────────────────────────────
    # VARIABLES
    # ─────────────────────────────────────────────────────

    # x[j,k] = 1 si job j est assigné à la position k
    x = {}
    for j in range(N):
        for k in range(N):
            x[j, k] = model.NewBoolVar(f'x_{j}_{k}')

    # S[k,m] : temps de début du job en position k sur machine m
    S = {}
    for k in range(N):
        for m in range(M):
            S[k, m] = model.NewIntVar(0, Cmax_bound, f'S_{k}_{m}')

    # C[k,m] : temps de fin du job en position k sur machine m
    C = {}
    for k in range(N):
        for m in range(M):
            C[k, m] = model.NewIntVar(0, Cmax_bound, f'C_{k}_{m}')

    # D[k] : due date du job en position k
    D = {}
    for k in range(N):
        D[k] = model.NewIntVar(0, Cmax_bound, f'D_{k}')

    # T[k] : tardiness du job en position k
    T = {}
    for k in range(N):
        T[k] = model.NewIntVar(0, Cmax_bound, f'T_{k}')

    # ─────────────────────────────────────────────────────
    # CONTRAINTES
    # ─────────────────────────────────────────────────────

    # C1 : chaque job assigné à exactement une position
    for j in range(N):
        model.AddExactlyOne(x[j, k] for k in range(N))

    # C2 : chaque position contient exactement un job
    for k in range(N):
        model.AddExactlyOne(x[j, k] for j in range(N))

    # C3 : C[k,m] = S[k,m] + processing time du job en position k
    for k in range(N):
        for m in range(M):
            p_expr = sum(x[j, k] * p[j][m] for j in range(N))
            model.Add(C[k, m] == S[k, m] + p_expr)

    # C4 : précédence des machines pour chaque position
    # job en position k doit finir sur machine m avant machine m+1
    for k in range(N):
        for m in range(M):
            # processing time du job en position k sur machine m
            p_km = model.NewIntVar(0, max_p, f'p_{k}_{m}')
            p_values = [p[j][m] for j in range(N)]
            # index = position du job dans la séquence
            pos_var = model.NewIntVar(0, N-1, f'pos_{k}')
            model.AddElement(pos_var, p_values, p_km)
            model.Add(C[k, m] == S[k, m] + p_km)

    # C5 : précédence des positions sur chaque machine
    # job en position k doit finir sur machine m avant que position k+1 commence
    for k in range(N - 1):
        for m in range(M):
            model.Add(C[k, m] <= S[k + 1, m])

    # C6 : D[k] = due date du job assigné à la position k
    for k in range(N):
        d_expr = sum(x[j, k] * d[j] for j in range(N))
        model.Add(D[k] == d_expr)

    # C7 : T[k] ≥ C[k, M-1] - D[k]  (tardiness)
    for k in range(N):
        model.Add(T[k] >= C[k, M - 1] - D[k])

    # C8 : T[k] ≥ 0
    for k in range(N):
        model.Add(T[k] >= 0)

    # ─────────────────────────────────────────────────────
    # FONCTION OBJECTIF : MIN ΣT[k]
    # ─────────────────────────────────────────────────────

    model.Minimize(sum(T[k] for k in range(N)))

    # ─────────────────────────────────────────────────────
    # RÉSOLUTION
    # ─────────────────────────────────────────────────────

    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers  = 4

    start_time  = time.time()
    status_code = solver.Solve(model)
    solve_time  = time.time() - start_time

    # ─────────────────────────────────────────────────────
    # EXTRACTION
    # ─────────────────────────────────────────────────────

    if status_code in [cp_model.OPTIMAL, cp_model.FEASIBLE]:

        # Reconstruire la séquence depuis les variables x[j,k]
        sequence = []
        for k in range(N):
            for j in range(N):
                if solver.Value(x[j, k]) == 1:
                    sequence.append(j)
                    break

        TT         = sum(solver.Value(T[k]) for k in range(N))
        obj_bound  = solver.BestObjectiveBound()
        gap        = round(abs(TT - obj_bound) / TT * 100, 2) if TT > 0 else 0.0
        status_str = "OPTIMAL" if status_code == cp_model.OPTIMAL else "FEASIBLE"

        print(f"  Status   : {status_str}")
        print(f"  Gap      : {gap}%")
        print(f"  Séquence : {[j+1 for j in sequence]}")
        print(f"  TT       : {TT}")
        print(f"  CPU      : {round(solve_time, 3)}s")

        # ── Sauvegarde CSV ────────────────────────────────
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            job_details = []
            for k in range(N):
                job      = sequence[k]
                cj       = solver.Value(C[k, M-1])
                tj       = solver.Value(T[k])
                dd       = d[job]
                start_m1 = solver.Value(S[k, 0])
                tardy    = "1" if tj > 0 else "0"

                job_details.append({
                    'job':      f"J{job+1}",
                    'due_date': dd,
                    'start_m1': start_m1,
                    'cj':       cj,
                    'tj':       tj,
                    'tardy':    tardy
                })

            with open(filepath, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "Job", "Due date", "Start M1",
                    "Completion Cj", "Tardiness", "Tardy"
                ])
                for d_row in job_details:
                    writer.writerow([
                        d_row['job'], d_row['due_date'], d_row['start_m1'],
                        d_row['cj'], d_row['tj'], d_row['tardy']
                    ])
                writer.writerow([])
                writer.writerow(["Status", "TT", "Gap(%)", "CPU(s)", "", ""])
                writer.writerow([status_str, TT, gap, round(solve_time, 3), "", ""])

            print(f"Résultats : {filepath}")

        return {
            'status':   status_str,
            'sequence': sequence,
            'TT':       TT,
            'gap':      gap,
            'cpu':      round(solve_time, 3)
        }

    else:
        print(f"  Pas de solution — status : {solver.StatusName(status_code)}")
        return None
    





def solve_milp_cmax(processing_times, due_dates, time_limit=None):

    pt = processing_times.T  # → (n_jobs x n_machines)
    N  = pt.shape[0]
    M  = pt.shape[1]
    p  = pt.tolist()
    d  = [int(due_dates[j]) for j in range(N)]


    Cmax_bound = int(np.sum(processing_times)) + 1

    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # ── VARIABLES ────────────────────────────────────────

    x = {}
    for j in range(N):
        for k in range(N):
            x[j, k] = model.NewBoolVar(f'x_{j}_{k}')

    S = {}
    C = {}
    for k in range(N):
        for m in range(M):
            S[k, m] = model.NewIntVar(0, Cmax_bound, f'S_{k}_{m}')
            C[k, m] = model.NewIntVar(0, Cmax_bound, f'C_{k}_{m}')
    
    T = {}
    for k in range(N):
        T[k] = model.NewIntVar(0, Cmax_bound, f'T_{k}')


    # ── CONTRAINTES ──────────────────────────────────────

    # C1 & C2
    for j in range(N):
        model.AddExactlyOne(x[j, k] for k in range(N))
    for k in range(N):
        model.AddExactlyOne(x[j, k] for j in range(N))

    # C3 : C[k,m] = S[k,m] + Σ_j x[j,k] * p[j][m]
    for k in range(N):
        for m in range(M):
            model.Add(
                C[k, m] == S[k, m] + sum(x[j, k] * p[j][m] for j in range(N))
            )

    # C4 : précédence machines
    for k in range(N):
        for m in range(M - 1):
            model.Add(C[k, m] <= S[k, m + 1])

    # C5 : précédence positions
    for k in range(N - 1):
        for m in range(M):
            model.Add(C[k, m] <= S[k + 1, m])
    
    # ← NOUVEAU C6 : T[k] >= C[k,M-1] - Σ_j x[j,k] * d[j]
    for k in range(N):
        d_k = sum(x[j, k] * d[j] for j in range(N))
        model.Add(T[k] >= C[k, M - 1] - d_k)
        model.Add(T[k] >= 0)

    # ── OBJECTIF : MIN Cmax = C[N-1, M-1] ───────────────
    model.Minimize(C[N - 1, M - 1])

    # ── RÉSOLUTION ───────────────────────────────────────
    if time_limit:
        solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers  = 4

    start_time  = time.time()
    status_code = solver.Solve(model)
    solve_time  = time.time() - start_time

    # ── EXTRACTION ───────────────────────────────────────
    if status_code in [cp_model.OPTIMAL, cp_model.FEASIBLE]:

        sequence = []
        for k in range(N):
            for j in range(N):
                if solver.Value(x[j, k]) == 1:
                    sequence.append(j)
                    break

        cmax_val   = solver.Value(C[N-1, M-1])
        obj_bound  = solver.BestObjectiveBound()
        gap        = round(abs(cmax_val - obj_bound) / cmax_val * 100, 2) if cmax_val > 0 else 0.0
        status_str = "OPTIMAL" if status_code == cp_model.OPTIMAL else "FEASIBLE"

        print(f"  Status : {status_str} | Gap : {gap}% | Cmax : {cmax_val} | CPU : {round(solve_time,3)}s")
        print(f"  Séquence : {[j+1 for j in sequence]}")

        return {
            'status':   status_str,
            'sequence': sequence,
            'Cmax':     cmax_val,
            'gap':      gap,
            'cpu':      round(solve_time, 3)
        }

    else:
        print(f"  Pas de solution — {solver.StatusName(status_code)}")
        return None