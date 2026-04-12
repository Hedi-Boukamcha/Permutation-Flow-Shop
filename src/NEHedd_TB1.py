import os
import csv
import time
import numpy as np

from src.dd_generator import generate_due_dates_brah
from src.scheduler import compute_completion_times
from src.data_loader import load_all

"""def compute_completion_times(sequence, processing_times):
    
    Calcule les temps de fin C_ij de chaque job j sur chaque machine i 
    pour la séquence donnée.

    n_jobs, n_machines = processing_times.shape
    completion_times = np.zeros((n_machines, n_jobs))
    
    for job in sequence:
        for machine in range(n_machines):
            if machine == 0:
                completion_times[machine, job] = (completion_times[machine, job-1] 
                                                  if job > 0 else 0) + processing_times[job, machine] 
            else:
                completion_times[machine, job] = max(completion_times[machine-1, job], 
                                                     completion_times[machine, job-1] if job > 0 else 0) + processing_times[job, machine]
    return completion_times"""


def compute_tardiness(sequence, processing_times, due_dates):
    """Calcul le retard total de la séquence"""
    completion_times = compute_completion_times(sequence, processing_times)
    tardiness = np.maximum(completion_times[-1] - due_dates[sequence], 0)
    return tardiness.sum() 


def compute_idle_time(sequence, processing_times):
    """Calcul l'idle time total IT1 de la séquence"""
    completion_times = compute_completion_times(sequence, processing_times)
    idle_time = completion_times.sum() - processing_times[:, sequence].sum()
    return idle_time


def NEHedd_IT1(processing_times, due_dates):
    """
    NEHedd avec départage des égalités par IT1
    """
    n_jobs = len(due_dates)
    
    # Trier les jobs par due dates croissantes
    job_order = np.argsort(due_dates)
    
    best_seq = []
    
    for i in range(n_jobs):
        min_tard = float('inf')
        cand_seqs = []
        
        for j in range(len(best_seq)+1):
            cand_seq = best_seq[:j] + [job_order[i]] + best_seq[j:]
            tard = compute_tardiness(cand_seq, processing_times, due_dates)
            
            if tard < min_tard:
                min_tard = tard
                cand_seqs = [cand_seq]
            elif tard == min_tard:
                cand_seqs.append(cand_seq)
        
        if len(cand_seqs) > 1:
            # Départager les égalités par IT1
            best_seq = min(cand_seqs, key=lambda seq: compute_idle_time(seq, processing_times))
        else:
            best_seq = cand_seqs[0]
            
    return best_seq




def results_nehedd_it1(datasets, output_dir='./results'):
    print(f"Exécution de NEHedd_IT1 avec enregistrement des résultats dans {output_dir}")
    
    for name, instances in datasets.items():
        print(f"\n  Dataset : {name} ({len(instances)} instances)")
        results = []
        
        for idx, inst in enumerate(instances):
            pt = inst['processing_times']
            due_dates = generate_due_dates_brah(inst, tau=2)
            
            start_time = time.time()
            sequence = NEHedd_IT1(pt, due_dates) 
            computing_time = time.time() - start_time
            
            tardiness = compute_tardiness(sequence, pt, due_dates)
            
            row = {
                "instance":   idx + 1,
                "n_jobs":     inst['n_jobs'],
                "n_machines": inst['n_machines'],
                "lb":         inst['lb'],
                "ub":         inst['ub'],
                "TT":         tardiness,
                "cpu_time_s": round(computing_time, 6),
                "sequence":   " ".join(str(j + 1) for j in sequence),
            }
            results.append(row)
            
            print(f"    Instance {idx+1:2d} | TT={tardiness:8d} | CPU={computing_time:.4f}s")
        
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"results_nehedd_it1_{name}.csv")
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['instance', 'n_jobs', 'n_machines', 'lb', 'ub', 'TT', 'cpu_time_s', 'sequence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)