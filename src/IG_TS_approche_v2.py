import numpy as np

from src.scheduler import compute_completion_times

"""# ─────────────────────────────────────────────
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

    return C"""


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


# ─────────────────────────────────────────────
# ITERATED GREEDY (FINAL)
# ─────────────────────────────────────────────
def ig_1F(pt, due_dates, weights=None, objective='TT',
          k=4, max_iter=100):

    # initialisation
    sequence = nehedd(pt, due_dates, weights, objective)

    best_seq = sequence[:]
    best_val = compute_objectives(sequence, pt, due_dates, weights)[objective]

    history = [best_val]

    print(f"IG start — {objective} initial = {best_val}")

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

        if new_val < best_val:
            best_seq = new_seq[:]
            best_val = new_val

        history.append(best_val)

        if (it+1) % 10 == 0:
            print(f"Iter {it+1} — {objective} = {best_val}")

    print(f"IG end — best {objective} = {best_val}")

    return best_seq, best_val, history


# ─────────────────────────────────────────────
# EXEMPLE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":

    np.random.seed(0)

    n_jobs = 20
    n_machines = 5

    pt = np.random.randint(1, 20, size=(n_machines, n_jobs))
    due_dates = np.random.randint(50, 200, size=n_jobs)

    best_seq, best_val, history = ig_1F(
        pt,
        due_dates,
        objective='TT',
        k=4,
        max_iter=50
    )

    print("\nBest sequence:", best_seq)
    print("Best TT:", best_val)