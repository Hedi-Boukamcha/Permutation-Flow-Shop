import time
from ortools.sat.python import cp_model
import numpy as np
import csv
import os

from src.scheduler import compute_objectives
from src.initial_solution import nehedd

def solve_milp_tt(processing_times, due_dates, time_limit,
                  filepath=None, instance_name=None, use_heuristic=None):
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


    if instance_name is None:
        instance_name = os.path.basename(filepath) if filepath else "unknown_instance"
    """horizontab = []
    for i in range(n_machines):
        total = sum(processing_times[i][j] for j in range(n_jobs))
        horizontab.append(total)
    horizon =  max(horizontab)
    L =1 * max(horizontab)"""
    horizon = 2*int(np.sum(processing_times))
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # ─────────────────────────────────────────────────────
    # VARIABLES
    # ─────────────────────────────────────────────────────

    S = {}
    for j in range(n_jobs):
        for i in range(n_machines):
            S[j, i] = model.NewIntVar(0, horizon, f"S_{j}_{i}")

    """# Pré-calcul des due dates moins la somme des processing times pour chaque job
    dds = {}
    for j in range(n_jobs):
        dds[j] = due_dates[j] - sum(processing_times[i][j] for i in range(n_machines))"""
    #
    """for j in range(n_jobs):
        for k in range(j + 1, n_jobs):
            if due_dates[k] < due_dates[j] and dds[k] <= dds[j]:
                model.Add(x[j, k] == 0)"""     

    # Création des variables de séquence
    """x = {}
    for j in range(n_jobs):
        for k in range(j+1, n_jobs):
            if dds[k] < dds[j]:
                # On ne crée pas x[j,k] car k ne peut pas être séquencé après j
                continue
            x[j, k] = model.NewBoolVar(f'x_{j}_{k}')"""
    
    min_start_m1 = model.NewIntVar(0, horizon, "min_start_m1")
    model.Add(min_start_m1 == 0)
              
    x = {}  
    for j in range(n_jobs):
        for k in range(j+1, n_jobs):
            x[j, k] = model.NewBoolVar(f"x_{j}_{k}")
    #
    """for j in range(n_jobs):
        for k in range(j+1, n_jobs):
            if dds[k] < dds[j]:
                model.Add(x[j, k] == 0)"""
    C = {}
    for j in range(n_jobs):
        C[j] = model.NewIntVar(0, horizon, f"C_{j}")

    T = {}    
    for j in range(n_jobs):
        T[j] = model.NewIntVar(0, horizon, f"T_{j}")

    model.AddMinEquality(min_start_m1, [S[j, 0] for j in range(n_jobs)])

    # ─────────────────────────────────────────────────────
    # CONTRAINTESdds
    # ─────────────────────────────────────────────────────
    # (1) x_jk = 1 ssi j précède k
    """for j in range(n_jobs):
        for k in range(j + 1, n_jobs):
            if (j, k) in x:
                model.Add(x[j, k] == 1).OnlyEnforceIf(x[j, k])
                model.Add(x[j, k] == 0).OnlyEnforceIf(x[j, k].Not())"""

    # (2) S_ji ≥ S_j,i-1 + p_j,i-1
     # (2) S_ji ≥ S_j,i-1 + p_j,i-1
    for j in range(n_jobs):
        for i in range(1, n_machines):
            model.Add(
                S[j, i] >= S[j, i-1] + int(processing_times[i-1][j])
            )

    # (3) Précédences conditionnelles
    for j in range(n_jobs):
        for k in range(j+1, n_jobs):
            if (j, k) in x:
                for i in range(n_machines):
                    model.Add(S[k,i] >= S[j,i] + processing_times[i][j]).OnlyEnforceIf(x[j,k])
                    model.Add(S[j,i] >= S[k,i] + processing_times[i][k]).OnlyEnforceIf(x[j,k].Not())

    # (4) C_j = S_j,m + p_j,m             
    for j in range(n_jobs):
        model.Add(C[j] == S[j, n_machines - 1] + int(processing_times[n_machines - 1][j]))

    # (5) T_j ≥ C_j - d_j
    for j in range(n_jobs):
        model.Add(T[j] >= C[j] - int(due_dates[j]))  
        model.Add(T[j] >= 0)
        
    # Borne inférieure sur les dates de fin
    for j in range(n_jobs):
        model.Add(C[j] >= sum(processing_times[i][j] for i in range(n_machines)))
    
    intervals = {}
    for j in range(n_jobs):
        for i in range(n_machines):
            intervals[j, i] = model.NewIntervalVar(
                S[j, i],
                int(processing_times[i][j]),
                S[j, i] + int(processing_times[i][j]),
                f"I_{j}_{i}"
            )
    for i in range(n_machines):
        model.AddNoOverlap([intervals[j, i] for j in range(n_jobs)])
    
    #
    sequence_vars = [x[j,k] for j in range(n_jobs) for k in range(j+1, n_jobs) if (j,k) in x]
    model.AddDecisionStrategy(sequence_vars, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

    # Calcul et injection de la solution heuristique
    heuristic_seq = None
    if use_heuristic:

            print(f"[HEUR] Calcul de la solution initiale NEHedd pour {instance_name}", flush=True)

            heuristic_seq = nehedd(
                processing_times=processing_times,
                due_dates=due_dates,
                weights=None,
                objective='TT'
            )

            print(f"[HEUR] Séquence heuristique = {[j + 1 for j in heuristic_seq]}", flush=True)
            #
            if heuristic_seq is not None and len(heuristic_seq) > 0:

                # Warm-start : hint sur les variables de séquence
                for j in range(n_jobs):
                    for k in range(j+1, n_jobs):
                        if (j, k) in x:
                            if heuristic_seq.index(j) < heuristic_seq.index(k):
                                model.AddHint(x[j, k], 1)
                            else:
                                model.AddHint(x[j, k], 0)
                current_time = 0
                for job in heuristic_seq:
                    model.AddHint(S[job, 0], current_time)
                    current_time += int(processing_times[0][job])
            else:
                print("[WARNING] heuristic_seq est vide ou None, pas de warm start.", flush=True)

    # ─────────────────────────────────────────────────────
    # OBJECTIF
    # ─────────────────────────────────────────────────────
    model.Minimize(sum(T[j] for j in range(n_jobs)))

    # ─────────────────────────────────────────────────────
    # PARAMÈTRES SOLVEUR
    # ─────────────────────────────────────────────────────
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 16
    solver.parameters.log_search_progress = True
    solver.parameters.log_to_stdout = True

    # ─────────────────────────────────────────────────────
    # RÉSOLUTION
    # ─────────────────────────────────────────────────────
    status = solver.Solve(model)
    validation_error = model.Validate()
    print("VALIDATION:", validation_error)

    # ─────────────────────────────────────────────────────
    # EXTRACTION DE LA SOLUTION
    # ─────────────────────────────────────────────────────
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:

        """job_starts = [(j, solver.Value(S[j, 0])) for j in range(n_jobs)]
        sequence = [j for j, _ in sorted(job_starts, key=lambda x: x[1])]"""
        #
        job_starts = [(j, solver.Value(S[j, 0]), solver.Value(C[j])) for j in range(n_jobs)]
        sequence = [j for j, _, _ in sorted(job_starts, key=lambda x: (x[1], x[2], x[0]))]

        TT = sum(solver.Value(T[j]) for j in range(n_jobs))
        status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        elapsed_time = solver.WallTime()
        objective_value = solver.ObjectiveValue()
        best_bound = solver.BestObjectiveBound()

        if objective_value != 0:
            gap = abs(objective_value - best_bound) / abs(objective_value) * 100
        else:
            gap = 0.0 if best_bound == 0 else float('inf')

        # Détails par job
        job_details = []
        for j in range(n_jobs):
            start_m1 = solver.Value(S[j, 0])
            cj = solver.Value(C[j])
            tj = solver.Value(T[j])
            tardy = 1 if tj > 0 else 0

            job_details.append({
                'job': j + 1,
                'due_date': int(due_dates[j]),
                'start_m1': start_m1,
                'cj': cj,
                'tj': tj,
                'tardy': tardy
            })

        result_table = []
        result_table.append(["Job", "Due date", "Start M1", "Completion Cj", "Tardiness", "Tardy"])

        details_by_job = {d["job"] - 1: d for d in job_details}
        for j in sequence:
            d = details_by_job[j]
            result_table.append([
                d['job'],
                d['due_date'],
                d['start_m1'],
                d['cj'],
                d['tj'],
                d['tardy']
            ])  

        result_table.append(["", "", "", "", "", ""])
        result_table.append(["Status", "TT", "Temps (s)", "Gap (%)", "BestBound", "Objective"])
        result_table.append([status_str, TT, f"{elapsed_time:.4f}", f"{gap:.2f}", best_bound, objective_value])

        print(f"[DONE] {instance_name} | status={status_str} | TT={TT} | gap={gap:.2f}% | time={elapsed_time:.2f}s", flush=True)

        # Sauvegarde détaillée par instance
        if filepath:
            output_dir = os.path.dirname(filepath)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(filepath, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(result_table)
                csvfile.flush()
                os.fsync(csvfile.fileno())

            print(f"[SAVE] Résultats détaillés sauvegardés dans : {filepath}", flush=True)

        return {
            'status': status_str,
            'sequence': sequence,
            'TT': TT,
            'time': elapsed_time,
            'gap': gap,
            'best_bound': best_bound,
            'objective_value': objective_value
        }

    else:
        elapsed_time = solver.WallTime()
        status_str = solver.StatusName(status)

        print(f"[FAIL] {instance_name} | status={status_str} | time={elapsed_time:.2f}s", flush=True)

        return {
            'status': status_str,
            'sequence': [],
            'TT': "",
            'time': elapsed_time,
            'gap': "",
            'best_bound': solver.BestObjectiveBound(),
            'objective_value': solver.ObjectiveValue()
        }