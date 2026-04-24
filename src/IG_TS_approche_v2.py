import csv
import os
import time
import numpy as np

#from src.scheduler import compute_completion_times

# ─────────────────────────────────────────────
# COMPLETION TIMES
# ─────────────────────────────────────────────
def compute_completion_times(sequence, pt):
    m = pt.shape[0]
    n = len(sequence)

    C = np.zeros((m, n), dtype=float)

    for j in range(n):
        job = sequence[j]
        for i in range(m):
            if i == 0 and j == 0:
                C[i][j] = pt[i][job]
            elif i == 0:
                C[i][j] = C[i][j-1] + pt[i][job]
            elif j == 0:
                C[i][j] = C[i-1][j] + pt[i][job]
            else:
                C[i][j] = max(C[i-1][j], C[i][j-1]) + pt[i][job]

    return C


# ─────────────────────────────────────────────
# OBJECTIVES
# ─────────────────────────────────────────────
def compute_objectives(sequence, pt, due_dates, weights=None):

    if weights is None:
        weights = np.ones(len(due_dates))

    C = compute_completion_times(sequence, pt)
    completion = C[-1]

    Tj = np.maximum(completion - due_dates[sequence], 0)

    TT   = np.sum(Tj)
    TWT  = np.sum(weights[sequence] * Tj)
    Tmax = np.max(Tj)
    NT   = np.sum(Tj > 0)

    return {
        'TT': TT,
        'TWT': TWT,
        'T_max': Tmax,
        'NT': NT,
        'Tj': Tj
    }


# ─────────────────────────────────────────────
# NEH-EDD INITIAL SOLUTION
# ─────────────────────────────────────────────
def nehedd(pt, due_dates, weights=None, objective='TT'):

    jobs = list(range(len(due_dates)))

    # tri EDD
    jobs_sorted = sorted(jobs, key=lambda j: due_dates[j])

    sequence = []

    for job in jobs_sorted:

        best_seq = None
        best_val = float('inf')

        for pos in range(len(sequence)+1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            val = compute_objectives(candidate, pt, due_dates, weights)[objective]

            if val < best_val:
                best_val = val
                best_seq = candidate

        sequence = best_seq

    return sequence


# ─────────────────────────────────────────────
# DESTRUCTION
# ─────────────────────────────────────────────
def destruction(sequence, pt, due_dates, weights, k):

    obj = compute_objectives(sequence, pt, due_dates, weights)
    Tj  = obj['Tj']

    positions = sorted(range(len(sequence)),
                       key=lambda i: Tj[i],
                       reverse=True)

    remove_idx = set(positions[:k])

    partial = [sequence[i] for i in range(len(sequence)) if i not in remove_idx]
    removed = [sequence[i] for i in positions[:k]]

    return partial, removed


# ─────────────────────────────────────────────
# RECONSTRUCTION
# ─────────────────────────────────────────────
def reconstruction(partial, removed, pt, due_dates, weights, objective):

    removed_sorted = sorted(removed, key=lambda j: due_dates[j])

    sequence = partial[:]

    for job in removed_sorted:

        best_seq = None
        best_val = float('inf')

        for pos in range(len(sequence)+1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            val = compute_objectives(candidate, pt, due_dates, weights)[objective]

            if val < best_val:
                best_val = val
                best_seq = candidate

        sequence = best_seq

    return sequence


# ─────────────────────────────────────────────
# LOCAL SEARCH (IMPORTANT)
# ─────────────────────────────────────────────
def local_search(sequence, pt, due_dates, weights, objective):

    improved = True

    while improved:
        improved = False

        for i in range(len(sequence)):
            job = sequence[i]
            partial = sequence[:i] + sequence[i+1:]

            best_seq = sequence
            best_val = compute_objectives(sequence, pt, due_dates, weights)[objective]

            for j in range(len(partial)+1):

                candidate = partial[:j] + [job] + partial[j:]
                val = compute_objectives(candidate, pt, due_dates, weights)[objective]

                if val < best_val:
                    best_val = val
                    best_seq = candidate
                    improved = True

            sequence = best_seq

    return sequence

def save_detailed_results(sequence, pt, due_dates, weights, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    C = compute_completion_times(sequence, pt)
    completion = C[-1]

    if weights is None:
        weights = np.ones(len(due_dates))

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Job", "Due date", "Weight", "Start M1", "Completion Cj", "Tardiness", "Tardy"])

        for j_idx, job in enumerate(sequence):

            start = 0 if j_idx == 0 else C[0][j_idx-1]
            cj = completion[j_idx]
            tardiness = max(cj - due_dates[job], 0)
            tardy = 1 if tardiness > 0 else 0

            writer.writerow([
                f"J{job+1}",
                due_dates[job],
                weights[job],
                start,
                cj,
                tardiness,
                tardy
            ])

        # Résumé
        TT = sum(max(completion[i] - due_dates[sequence[i]], 0) for i in range(len(sequence)))
        TWT = sum(weights[sequence[i]] * max(completion[i] - due_dates[sequence[i]], 0) for i in range(len(sequence)))
        Tmax = max(max(completion[i] - due_dates[sequence[i]], 0) for i in range(len(sequence)))
        NT = sum(1 for i in range(len(sequence)) if completion[i] > due_dates[sequence[i]])

        writer.writerow([])
        writer.writerow(["TT", "TWT", "T_max", "NT"])
        writer.writerow([TT, TWT, Tmax, NT])


def IG_1F(instance, due_dates, weights=None, objective='TT',
          k=4, max_iter=10, filepath=None, verbose=True):

    start = time.time()

    pt = instance['processing_times']

    # ─────────────────────────────
    # INITIALISATION (NEH)
    # ─────────────────────────────
    sequence = nehedd(pt, due_dates, weights, objective)

    best_seq = sequence[:]
    best_val = compute_objectives(sequence, pt, due_dates, weights)[objective]
    obj = compute_objectives(best_seq, pt, due_dates, weights)
    history = [best_val]

    if verbose:
        print(f"IG start — {objective} initial = {best_val}")

    # ─────────────────────────────
    # BOUCLE PRINCIPALE
    # ─────────────────────────────
    for it in range(max_iter):

        # destruction
        partial, removed = destruction(sequence, pt, due_dates, weights, k)

        # reconstruction
        new_seq = reconstruction(partial, removed, pt, due_dates, weights, objective)

        # local search
        new_seq = local_search(new_seq, pt, due_dates, weights, objective)

        new_val = compute_objectives(new_seq, pt, due_dates, weights)[objective]

        # acceptation
        if new_val <= best_val:
            sequence = new_seq[:]

        # mise à jour best
        if new_val < best_val:
            best_seq = new_seq[:]
            best_val = new_val

        history.append(best_val)

        if verbose and (it+1) % 10 == 0:
            print(f"  Iter {it+1} — {objective} = {best_val}")

    elapsed = time.time() - start

    if verbose:
        print(f"IG end — best {objective} = {best_val}")

    # ─────────────────────────────
    # SAUVEGARDE
    # ─────────────────────────────
    if filepath:
        save_detailed_results(
        sequence=best_seq,
        pt=pt,
        due_dates=due_dates,
        weights=weights,
        filepath=filepath
    )

    # ─────────────────────────────
    # FORMAT STANDARD (comme MILP)
    # ─────────────────────────────
    return {
        'sequence': best_seq,
        'TT': obj['TT'],
        'TWT': obj['TWT'],
        'T_max': obj['T_max'],
        'NT': obj['NT'],
        'time': elapsed,
        'history': history
    }


