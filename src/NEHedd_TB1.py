import os
import csv
import time
import numpy as np

from src.results import save_results
from src.dd_generator import generate_due_dates_brah
from src.scheduler import compute_completion_times, compute_objectives
from src.data_loader import load_instance


# ─────────────────────────────────────────────
# 🔧 Forcer format (machines, jobs)
# ─────────────────────────────────────────────
def ensure_pt_format(pt):
    if pt.shape[0] > pt.shape[1]:
        return pt.T
    return pt


# ─────────────────────────────────────────────
# 📌 Tardiness
# ─────────────────────────────────────────────
def compute_tardiness(sequence, pt, due_dates):
    pt = ensure_pt_format(pt)

    C = compute_completion_times(sequence, pt)
    Cm = C[-1]

    return np.maximum(Cm - due_dates[sequence], 0).sum()


# ─────────────────────────────────────────────
# 📌 Idle Time (IT1)
# ─────────────────────────────────────────────
def compute_idle_time(sequence, pt):
    pt = ensure_pt_format(pt)

    C = compute_completion_times(sequence, pt)

    idle_time = 0
    for i in range(pt.shape[0]):
        idle_time += C[i][-1] - pt[i, sequence].sum()

    return idle_time


# ─────────────────────────────────────────────
# 📌 Insertion
# ─────────────────────────────────────────────
def evaluate_insertion(partial_seq, job, pt, due_dates, weights=None, objective='TT'):

    best_value = None
    best_it1   = None
    best_pos   = 0
    ties_count = 0

    for pos in range(len(partial_seq) + 1):

        candidate = partial_seq[:pos] + [job] + partial_seq[pos:]

        value = compute_objectives(
            candidate,
            pt,
            due_dates,
            weights=weights,
        )[objective]

        if best_value is None or value < best_value:
            best_value = value
            best_it1   = compute_idle_time(candidate, pt)
            best_pos   = pos
            ties_count = 1

        elif value == best_value:
            ties_count += 1
            it1 = compute_idle_time(candidate, pt)

            if it1 < best_it1:
                best_it1 = it1
                best_pos = pos

    return best_pos, ties_count


# ─────────────────────────────────────────────
# 📌 NEHedd + IT1
# ─────────────────────────────────────────────
"""def NEHedd_IT1(pt, due_dates):

    pt = ensure_pt_format(pt)
    n_jobs = pt.shape[1]

    t_start = time.perf_counter()

    edd_order = np.argsort(due_dates, kind='stable')

    sequence   = [edd_order[0]]
    total_ties = 0

    for k in range(1, n_jobs):
        job = edd_order[k]

        best_pos, ties = evaluate_insertion(sequence, job, pt, due_dates)

        total_ties += max(0, ties - 1)
        sequence.insert(best_pos, job)

    elapsed = time.perf_counter() - t_start

    return sequence, total_ties, elapsed"""

def NEHedd_IT1(pt, due_dates, weights=None, objective='TT'):
    pt = ensure_pt_format(pt)
    n_jobs = pt.shape[1]

    if weights is None:
        weights = np.ones(n_jobs, dtype=int)

    t_start = time.perf_counter()

    # base EDD
    edd_order = np.argsort(due_dates, kind='stable')

    sequence   = [edd_order[0]]
    total_ties = 0

    for k in range(1, n_jobs):
        job = edd_order[k]

        best_pos, ties = evaluate_insertion(
            sequence,
            job,
            pt,
            due_dates,
            weights=weights,
            objective=objective
        )

        total_ties += max(0, ties - 1)
        sequence.insert(best_pos, job)

    elapsed = time.perf_counter() - t_start

    return sequence, total_ties, elapsed


# ─────────────────────────────────────────────
# 📌 Pipeline
# ─────────────────────────────────────────────
def results_nehedd_it1(instances_dir, output_dir='./results'):

    print(f"Exécution de NEHedd_IT1")

    for subdir in ['20j_5m', '50j_10m']:

        print(f"\nDataset : {subdir}")
        results = []

        instance_dir = os.path.join(instances_dir, subdir)
        instance_files = sorted(os.listdir(instance_dir))

        for idx, instance_file in enumerate(instance_files):

            instance_path = os.path.join(instance_dir, instance_file)
            instance = load_instance(instance_path)

            pt = instance['processing_times']
            due_dates = instance['due_date']

            start_time = time.time()

            sequence, total_ties, elapsed = NEHedd_IT1(pt, due_dates)

            computing_time = time.time() - start_time

            tardiness = compute_tardiness(sequence, pt, due_dates)

            results.append({
                'instance_id': f"{subdir}_{idx+1}",
                'objective': int(tardiness),
                'computing_time': computing_time
            })

            print(f"Instance {subdir}_{idx+1} | TT={tardiness} | Time={computing_time:.2f}s")

        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"results_nehedd_it1_{subdir}.csv")

        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['instance_id', 'objective', 'computing_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)

def run_nehedd_it1(instance, weights=None, objective='TT', filepath=None):
    pt = instance['processing_times']
    due_dates = instance['due_date']

    if weights is None:
        weights = np.ones(pt.shape[1], dtype=int)

    start = time.time()

    sequence, total_ties, elapsed = NEHedd_IT1(
        pt,
        due_dates,
        weights=weights,
        objective=objective
    )

    obj = compute_objectives(sequence, pt, due_dates, weights)
    total_time = time.time() - start

    if filepath:
        save_results(
            sequence=sequence,
            processing_times=pt,
            due_dates=due_dates,
            weights=weights,
            filepath=filepath
        )

    return {
        "sequence": sequence,
        "TT": obj["TT"],
        "TWT": obj["TWT"],
        "T_max": obj["T_max"],
        "NT": obj["NT"],
        "time": total_time,
        "total_ties": total_ties
    }