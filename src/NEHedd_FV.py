import os
import csv
import time
import numpy as np

from src.scheduler    import compute_completion_times
from src.dd_generator import generate_due_dates_brah
from src.data_loader  import load_all


# ─────────────────────────────────────────────
# 🔧 Forcer format (machines, jobs)
# ─────────────────────────────────────────────
def ensure_pt_format(pt):
    pt = np.asarray(pt)
    if pt.ndim != 2:
        raise ValueError("processing_times doit être une matrice 2D")
    if pt.shape[0] > pt.shape[1]:
        return pt.T
    return pt


# ─── TT ──────────────────────────────────────
def compute_tt(sequence, pt, due_dates):

    pt = ensure_pt_format(pt)
    sequence = np.asarray(sequence, dtype=int)
    due_dates = np.asarray(due_dates)

    C  = compute_completion_times(sequence.tolist(), pt)
    Cm = C[-1]

    return int(np.maximum(Cm - due_dates[sequence], 0).sum())


# ─── IT1 ─────────────────────────────────────
def compute_it1(sequence, pt):

    pt = ensure_pt_format(pt)
    sequence = np.asarray(sequence, dtype=int)

    C = compute_completion_times(sequence.tolist(), pt)

    it1 = 0
    for i in range(pt.shape[0]):
        # ⚡ vectorisé + safe
        sum_p = int(pt[i, sequence].sum())
        it1  += int(C[i][-1]) - sum_p

    return int(it1)


# ─── Insertion ───────────────────────────────
def evaluate_insertion(partial_seq, job, pt, due_dates):

    best_tt    = None
    best_it1   = None
    best_pos   = 0
    ties_count = 0

    for pos in range(len(partial_seq) + 1):

        candidate = partial_seq[:pos] + [job] + partial_seq[pos:]

        tt = compute_tt(candidate, pt, due_dates)

        if best_tt is None or tt < best_tt:
            best_tt    = tt
            best_it1   = compute_it1(candidate, pt)
            best_pos   = pos
            ties_count = 1

        elif tt == best_tt:
            ties_count += 1
            it1 = compute_it1(candidate, pt)

            if it1 < best_it1:
                best_it1 = it1
                best_pos = pos

    return best_pos, ties_count


# ─── NEHedd ──────────────────────────────────
def nehedd_tbit1(instance, due_dates):

    # 🔥 accès sécurisé
    pt = ensure_pt_format(instance['processing_times'])
    due_dates = np.asarray(due_dates)

    # 🔥 robuste (au cas où n_jobs absent)
    n_jobs = instance.get('n_jobs', pt.shape[1])

    if len(due_dates) != n_jobs:
        raise ValueError("Mismatch entre due_dates et nombre de jobs")

    t_start = time.perf_counter()

    edd_order = np.argsort(due_dates, kind='stable')

    sequence   = [int(edd_order[0])]
    total_ties = 0

    for k in range(1, n_jobs):

        job = int(edd_order[k])

        best_pos, ties = evaluate_insertion(sequence, job, pt, due_dates)

        total_ties += max(0, ties - 1)
        sequence.insert(best_pos, job)

    elapsed = time.perf_counter() - t_start

    return sequence, total_ties, elapsed


def run_nehedd_FV(name, instances):

    results = []
    print(f"\n  Dataset : {name} ({len(instances)} instances)")

    for idx, inst in enumerate(instances):

        # 🔥 correction accès
        pt        = inst['processing_times']
        due_dates = inst['due_date']

        # 🔧 sécuriser format (important)
        if pt.shape[0] > pt.shape[1]:
            pt = pt.T

        start_time = time.time()

        sequence, ties, elapsed = nehedd_tbit1(inst, due_dates)

        computing_time = time.time() - start_time

        tt = compute_tt(sequence, pt, due_dates)

        row = {
            "instance":   idx + 1,
            "n_jobs":     inst.get('n_jobs', pt.shape[1]),
            "n_machines": inst.get('n_machines', pt.shape[0]),
            "lb":         inst.get('lb', None),
            "ub":         inst.get('ub', None),
            "TT":         int(tt),
            "total_ties": ties,
            "cpu_time_s": round(computing_time, 6),
            "sequence":   " ".join(str(j + 1) for j in sequence),
        }

        results.append(row)

        print(f"    Instance {idx+1:2d} | TT={tt:8d} | Ties={ties:4d} | CPU={computing_time:.4f}s")

    return results