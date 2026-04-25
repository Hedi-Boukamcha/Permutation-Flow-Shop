from ortools.sat.python import cp_model
import numpy as np
import time

def solve(processing_times, due_dates, time_limit, 
                filepath=None, instance_name=None, use_heuristic=None):

    start_time = time.time()
    
    n_machines = processing_times.shape[0]
    n_jobs = processing_times.shape[1]
    
    model = cp_model.CpModel()
    
    horizon = processing_times.sum()
    
    # Variables
    start_times = {}
    end_times = {}
    tardiness = {}
    
    for j in range(n_jobs):
        for i in range(n_machines):
            start_times[j, i] = model.NewIntVar(0, horizon, f"start_{j}_{i}")
            end_times[j, i] = model.NewIntVar(0, horizon, f"end_{j}_{i}")
        tardiness[j] = model.NewIntVar(0, horizon, f"tardiness_{j}")
    
    rank = {}
    for j in range(n_jobs):
        for l in range(n_jobs):
            rank[j, l] = model.NewBoolVar(f"rank_{j}_{l}")

    # Contraintes
    for i in range(n_machines):
        for j in range(n_jobs):
            model.Add(end_times[j, i] == start_times[j, i] + int(processing_times[i, j]))
            if i > 0:
                model.Add(start_times[j, i] >= end_times[j, i-1])
        
    for j in range(n_jobs):
        model.Add(tardiness[j] >= end_times[j, n_machines-1] - int(due_dates[j]))
        
    for l in range(n_jobs):
        model.Add(sum(rank[j, l] for j in range(n_jobs)) == 1)
        
    for j in range(n_jobs):
        model.Add(sum(rank[j, l] for l in range(n_jobs)) == 1)
        
    for i in range(n_machines):
        for l in range(1, n_jobs):
            for j1 in range(n_jobs):
                for j2 in range(n_jobs):
                    if j1 != j2:
                        model.Add(start_times[j2, i] >= end_times[j1, i]).OnlyEnforceIf(rank[j1, l-1], rank[j2, l])
                        
    # Objectif
    model.Minimize(sum(tardiness[j] for j in range(n_jobs)))
    
    # Résolution
    solver = cp_model.CpSolver()
    if use_heuristic:
        solver.parameters.max_time_in_seconds = min(2 * n_jobs * n_machines / 1000, time_limit)
        solution = solver.SolveWithSolutionCallback(model, cp_model.ObjectiveSolutionPrinter())
    else:
        solver.parameters.max_time_in_seconds = time_limit
        solution = solver.Solve(model)
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 1
        solver.parameters.log_search_progress = True
        solver.parameters.log_to_stdout = True

    status = solver.StatusName()
    
    if solution:
        # Extraire la séquence
        sequence = []
        for l in range(n_jobs):
            for j in range(n_jobs):
                if solver.Value(rank[j, l]) == 1:
                    sequence.append(j)
                    break
        
        # Calculer le retard total 
        total_tardiness = sum(max(solver.Value(end_times[j, n_machines-1]) - due_dates[j], 0) for j in range(n_jobs))
                
        elapsed_time = solver.WallTime()
        
        # Calculer le gap si une solution optimale n'a pas été trouvée
        if status == cp_model.OPTIMAL:
            gap = 0
        elif status == cp_model.FEASIBLE:
            best_bound = solver.ObjectiveValue()
            gap = 100 * (total_tardiness - best_bound) / best_bound
        else:
            gap = None
        
        return {"sequence"        : np.array(sequence),
                "TT"              : total_tardiness, 
                "time"            : elapsed_time,
                "status"          : status,
                "gap"             : gap,
                "best_bound"      : solver.ObjectiveValue() if status != cp_model.OPTIMAL else total_tardiness,
                "objective_value" : total_tardiness}
    else:
        return {"sequence"        : None,
                "TT"              : None, 
                "time"            : None,
                "status"          : status,
                "gap"             : None,
                "best_bound"      : None,
                "objective_value" : None}