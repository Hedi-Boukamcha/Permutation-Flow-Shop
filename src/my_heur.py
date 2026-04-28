import os
import time
import math
import random
import numpy as np

from src.dd_generator import generate_weights, load_weights, save_weights
from src.scheduler import compute_completion_times, compute_objectives


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_detailed_results(sequence, pt, due_dates, weights, filepath, extra_summary=None):
    import csv

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if weights is None:
        weights = np.ones(len(due_dates), dtype=int)

    obj = compute_objectives(sequence, pt, due_dates, weights)
    C = compute_completion_times(sequence, pt)
    Cj = obj['Cj']
    Tj = obj['Tj']
    Uj = obj['Uj']

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["Job", "Due date", "Weight", "Start M1", "Completion Cj", "Tardiness", "Tardy"])

        for pos, job in enumerate(sequence):
            start_m1 = 0 if pos == 0 else C[0, pos - 1]
            writer.writerow([
                f"J{job+1}",
                int(due_dates[job]),
                int(weights[job]),
                int(start_m1),
                int(Cj[pos]),
                int(Tj[pos]),
                int(Uj[pos])
            ])

        writer.writerow([])
        writer.writerow(["TT", "TWT", "T_max", "NT"])
        writer.writerow([obj['TT'], obj['TWT'], obj['T_max'], obj['NT']])

        if extra_summary:
            writer.writerow([])
            writer.writerow(["Metric", "Value"])
            for k, v in extra_summary.items():
                writer.writerow([k, v])


# ─────────────────────────────────────────────
# JOB FEATURES / CRITICALITY
# ─────────────────────────────────────────────
def compute_job_features(pt, due_dates, weights=None):
    n_jobs = pt.shape[1]

    if weights is None:
        weights = np.ones(n_jobs, dtype=float)

    P = np.sum(pt, axis=0)
    due = due_dates.astype(float)

    p_min, p_max = np.min(P), np.max(P)
    d_min, d_max = np.min(due), np.max(due)
    w_min, w_max = np.min(weights), np.max(weights)

    def norm(x, a, b):
        if b - a < 1e-12:
            return np.zeros_like(x, dtype=float)
        return (x - a) / (b - a)

    Pn = norm(P, p_min, p_max)
    Dn = norm(due, d_min, d_max)
    Wn = norm(weights.astype(float), w_min, w_max)

    avg_p = np.mean(P) if np.mean(P) > 0 else 1.0
    slack = due - P
    s_min, s_max = np.min(slack), np.max(slack)
    Sn = norm(slack, s_min, s_max)

    features = []
    for j in range(n_jobs):
        features.append({
            "job": j,
            "P": float(P[j]),
            "due": float(due[j]),
            "weight": float(weights[j]),
            "slack": float(slack[j]),
            "Pn": float(Pn[j]),
            "Dn": float(Dn[j]),
            "Wn": float(Wn[j]),
            "Sn": float(Sn[j]),
            "urgency_score": float((1 - Dn[j]) + (1 - Sn[j])),
            "weighted_urgency": float((1 - Dn[j]) + (1 - Sn[j]) + Wn[j])
        })

    return features


def criticality_score(job, pt, due_dates, weights=None):
    if weights is None:
        w = 1.0
    else:
        w = float(weights[job])

    Pj = float(np.sum(pt[:, job]))
    dj = float(due_dates[job])

    # score élevé = job critique
    # petit due date + faible slack + grand temps + grand poids
    slack = dj - Pj
    return (1.8 / max(dj, 1.0)) + (1.2 / max(slack + abs(min(slack, 0)) + 1.0, 1.0)) + 0.02 * Pj + 0.15 * w


# ─────────────────────────────────────────────
# AUGMENTED INSERTION SCORE
# ─────────────────────────────────────────────
def objective_value(obj_dict, objective):
    return obj_dict[objective]


def augmented_score(sequence, pt, due_dates, weights, objective,
                    alpha=1.0, beta=0.15, gamma=0.10, delta=0.10):
    obj = compute_objectives(sequence, pt, due_dates, weights)

    # objectif principal
    main = obj[objective]

    # termes auxiliaires pour guider la recherche
    score = alpha * main

    if objective != 'TT':
        score += beta * obj['TT']
    if objective != 'TWT':
        score += gamma * obj['TWT']
    if objective != 'T_max':
        score += delta * obj['T_max']

    return score


# ─────────────────────────────────────────────
# CONSTRUCTIVE INSERTION
# ─────────────────────────────────────────────
def greedy_insert_from_order(order, pt, due_dates, weights, objective):
    seq = []

    for job in order:
        best_seq = None
        best_score = float('inf')

        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [job] + seq[pos:]
            score = augmented_score(cand, pt, due_dates, weights, objective)

            if score < best_score:
                best_score = score
                best_seq = cand

        seq = best_seq

    return seq


def generate_initial_orders(pt, due_dates, weights=None):
    n_jobs = pt.shape[1]
    jobs = list(range(n_jobs))

    if weights is None:
        weights = np.ones(n_jobs, dtype=float)

    P = np.sum(pt, axis=0)
    crit = {j: criticality_score(j, pt, due_dates, weights) for j in jobs}

    orders = []

    # EDD
    orders.append(sorted(jobs, key=lambda j: due_dates[j]))

    # SPT total
    orders.append(sorted(jobs, key=lambda j: P[j]))

    # weighted urgency
    orders.append(sorted(jobs, key=lambda j: (due_dates[j] / max(weights[j], 1e-9), P[j])))
    
    # NEH
    neh_order = []
    remaining_jobs = jobs[:]
    
    while remaining_jobs:
        best_job = None
        best_tt = float('inf')
        
        for job in remaining_jobs:
            cand_seq = neh_order + [job]
            tt = compute_objectives(cand_seq, pt, due_dates, weights)['TT']
            if tt < best_tt:
                best_job = job
                best_tt = tt
        
        neh_order.append(best_job)
        remaining_jobs.remove(best_job)
    
    # NEHedd
    nehedd_order = []
    remaining_jobs = sorted(jobs, key=lambda j: due_dates[j])
    
    while remaining_jobs:
        best_job = None
        best_tt = float('inf')
        
        for job in remaining_jobs:
            cand_seq = nehedd_order + [job]
            tt = compute_objectives(cand_seq, pt, due_dates, weights)['TT']
            if tt < best_tt:
                best_job = job
                best_tt = tt
        
        nehedd_order.append(best_job)
        remaining_jobs.remove(best_job)
    
    orders.append(nehedd_order)
    
    orders.append(neh_order)

    # criticality descending
    #orders.append(sorted(jobs, key=lambda j: crit[j], reverse=True))

    # hybrid due + processing + weight
    #orders.append(sorted(jobs, key=lambda j: (due_dates[j], P[j], weights[j])))

    return orders


def multi_start_initial_solution(pt, due_dates, weights=None, objective='TT'):
    orders = generate_initial_orders(pt, due_dates, weights)

    best_seq = None
    best_obj = None
    best_val = float('inf')

    for order in orders:
        seq = greedy_insert_from_order(order, pt, due_dates, weights, objective)
        obj = compute_objectives(seq, pt, due_dates, weights)
        val = objective_value(obj, objective)

        if val < best_val:
            best_val = val
            best_seq = seq
            best_obj = obj

    return best_seq, best_obj


"""# ─────────────────────────────────────────────
# DESTRUCTION
# ─────────────────────────────────────────────
def destruction_critical(sequence, pt, due_dates, weights, objective, k=4, rng=None):
    if rng is None:
        rng = random.Random()

    obj = compute_objectives(sequence, pt, due_dates, weights)
    tardiness_by_pos = obj['Tj']

    scored = []
    for pos, job in enumerate(sequence):
        score = 0.0
        score += 2.0 * tardiness_by_pos[pos]
        score += 40.0 * criticality_score(job, pt, due_dates, weights)
        score += 0.2 * np.sum(pt[:, job])
        scored.append((score, pos, job))

    scored.sort(reverse=True, key=lambda x: x[0])

    nb_critical = max(1, math.ceil(k * 0.7))
    chosen_positions = {pos for _, pos, _ in scored[:nb_critical]}

    remaining_positions = [i for i in range(len(sequence)) if i not in chosen_positions]
    rng.shuffle(remaining_positions)

    for pos in remaining_positions[:max(0, k - len(chosen_positions))]:
        chosen_positions.add(pos)

    partial = [sequence[i] for i in range(len(sequence)) if i not in chosen_positions]
    removed = [sequence[i] for i in range(len(sequence)) if i in chosen_positions]

    return partial, removed"""

# ─────────────────────────────────────────────
# DESTRUCTION
# ─────────────────────────────────────────────
def destruction_random(sequence, d, rng=None):
    if rng is None:
        rng = random.Random()

    indices = rng.sample(range(len(sequence)), d)
    removed = [sequence[i] for i in sorted(indices)]
    remaining = [j for i, j in enumerate(sequence) if i not in indices]

    return remaining, removed


# ─────────────────────────────────────────────
# RECONSTRUCTION
# ─────────────────────────────────────────────
"""def reconstruction_impact(partial, removed, pt, due_dates, weights, objective):
    # on réinsère d'abord les jobs les plus critiques
    removed_sorted = sorted(
        removed,
        key=lambda j: criticality_score(j, pt, due_dates, weights),
        reverse=True
    )

    seq = partial[:]

    for job in removed_sorted:
        best_seq = None
        best_score = float('inf')

        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [job] + seq[pos:]
            score = augmented_score(cand, pt, due_dates, weights, objective)

            if score < best_score:
                best_score = score
                best_seq = cand

        seq = best_seq

    return seq"""

def reconstruction_greedy(remaining, removed, pt, due_dates, weights=None, objective='TT'):
    seq = list(remaining)
    for job in removed:
        best_tt = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            tt = compute_objectives(candidate, pt, due_dates, weights)['TT']
            if tt < best_tt:
                best_tt = tt
                best_pos = pos
        seq.insert(best_pos, job)
    return seq

"""
# ─────────────────────────────────────────────
# TARGETED LOCAL SEARCH
# ─────────────────────────────────────────────
def targeted_jobs(sequence, pt, due_dates, weights, max_jobs=6):
    obj = compute_objectives(sequence, pt, due_dates, weights)
    Tj = obj['Tj']

    scored = []
    for pos, job in enumerate(sequence):
        score = 3.0 * Tj[pos] + 25.0 * criticality_score(job, pt, due_dates, weights)
        scored.append((score, pos, job))

    scored.sort(reverse=True, key=lambda x: x[0])

    jobs = []
    seen = set()
    for _, _, job in scored:
        if job not in seen:
            jobs.append(job)
            seen.add(job)
        if len(jobs) >= max_jobs:
            break

    return jobs


def local_search_targeted(sequence, pt, due_dates, weights, objective, max_rounds=10):
    current = sequence[:]
    current_val = compute_objectives(current, pt, due_dates, weights)[objective]

    improved = True
    rounds = 0

    while improved and rounds < max_rounds:
        improved = False
        rounds += 1

        candidate_jobs = targeted_jobs(current, pt, due_dates, weights, max_jobs=min(8, len(current)))

        for job in candidate_jobs:
            i = current.index(job)
            partial = current[:i] + current[i+1:]

            best_seq = current
            best_val = current_val

            for j in range(len(partial) + 1):
                if j == i:
                    continue

                cand = partial[:j] + [job] + partial[j:]
                val = compute_objectives(cand, pt, due_dates, weights)[objective]

                if val < best_val:
                    best_val = val
                    best_seq = cand

            if best_val < current_val:
                current = best_seq
                current_val = best_val
                improved = True

    return current



# ─────────────────────────────────────────────
# ELITE POOL
# ─────────────────────────────────────────────
def add_to_elite(elite_pool, sequence, pt, due_dates, weights, objective, max_size=5):
    val = compute_objectives(sequence, pt, due_dates, weights)[objective]

    # évite doublons
    seq_tuple = tuple(sequence)
    for old_seq, old_val in elite_pool:
        if tuple(old_seq) == seq_tuple:
            return elite_pool

    elite_pool.append((sequence[:], val))
    elite_pool.sort(key=lambda x: x[1])

    return elite_pool[:max_size]


def path_relinking(seq_a, seq_b, pt, due_dates, weights, objective):
    current = seq_a[:]
    best = current[:]
    best_val = compute_objectives(best, pt, due_dates, weights)[objective]

    for target_pos in range(len(seq_b)):
        wanted_job = seq_b[target_pos]
        current_pos = current.index(wanted_job)

        if current_pos != target_pos:
            job = current[current_pos]
            partial = current[:current_pos] + current[current_pos+1:]
            current = partial[:target_pos] + [job] + partial[target_pos:]

            val = compute_objectives(current, pt, due_dates, weights)[objective]
            if val < best_val:
                best = current[:]
                best_val = val

    return best


# ─────────────────────────────────────────────
# MAIN HEURISTIC
# ─────────────────────────────────────────────
def heuristic_due_date_pfsp(instance,
                            subdir=None,
                            weights_dir="data/weights",
                            weights=None,
                            objective='TT',
                            k=4,
                            max_iter=100,
                            seed=42,
                            elite_size=5,
                            filepath=None,
                            verbose=True):
    rng = random.Random(seed)
    np.random.seed(seed)

    pt = instance['processing_times']
    due_dates = instance['due_date']
    n_jobs = instance['n_jobs']
    
    
    # WEIGHTS MANAGEMENT
    # ─────────────────────────────────────────────
    if weights is None:

        if subdir is not None:
            weights_path = os.path.join(weights_dir, f"{subdir}_weights.csv")

            if os.path.exists(weights_path):
                weights = load_weights(weights_path)
                if verbose:
                    print(f"Weights chargés depuis {weights_path}", flush=True)
            else:
                weights = generate_weights(n_jobs, seed=seed)
                save_weights(weights, weights_path)
                if verbose:
                    print(f"Weights générés et sauvegardés dans {weights_path}", flush=True)
        else:
            weights = generate_weights(n_jobs, seed=seed)
            if verbose:
                print("Weights générés (pas de subdir fourni)", flush=True)



    start = time.time()

    # 1) initialisation multi-règles
    current_seq, current_obj = multi_start_initial_solution(pt, due_dates, weights, objective)
    current_val = current_obj[objective]

    best_seq = current_seq[:]
    best_val = current_val

    elite_pool = []
    elite_pool = add_to_elite(elite_pool, current_seq, pt, due_dates, weights, objective, max_size=elite_size)

    history = [best_val]

    if verbose:
        print(f"Heuristic start — objective={objective} initial={best_val}")

    no_improve = 0

    for it in range(max_iter):
        # 2) destruction ciblée
        remaining, removed = destruction_random(current_seq, rng=rng)

        # 3) reconstruction orientée impact
        new_seq = reconstruction_impact(remaining, removed, pt, due_dates, weights, objective)

        # 4) local search ciblée
        new_seq = local_search_targeted(new_seq, pt, due_dates, weights, objective, max_rounds=8)

        # 5) intensification par elite / path relinking de temps en temps
        if elite_pool and (it + 1) % 10 == 0:
            ref_seq = rng.choice(elite_pool)[0]
            pr_seq = path_relinking(new_seq, ref_seq, pt, due_dates, weights, objective)
            pr_val = compute_objectives(pr_seq, pt, due_dates, weights)[objective]
            new_val = compute_objectives(new_seq, pt, due_dates, weights)[objective]

            if pr_val < new_val:
                new_seq = pr_seq

        new_obj = compute_objectives(new_seq, pt, due_dates, weights)
        new_val = new_obj[objective]

        # 6) acceptation
        if new_val < current_val:
            current_seq = new_seq[:]
            current_val = new_val
            no_improve = 0
        else:
            # acceptation faible pour diversification
            if rng.random() < 0.10:
                current_seq = new_seq[:]
                current_val = new_val
            no_improve += 1

        # 7) update best / elite
        if new_val < best_val:
            best_seq = new_seq[:]
            best_val = new_val
            no_improve = 0

        elite_pool = add_to_elite(elite_pool, new_seq, pt, due_dates, weights, objective, max_size=elite_size)
        history.append(best_val)

        # 8) diversification si stagnation
        if no_improve >= 15:
            k_big = min(max(k + 2, 5), len(current_seq) // 2 if len(current_seq) > 2 else 1)
            partremainingial, removed = destruction_random(current_seq, rng=rng)
            current_seq = reconstruction_impact(remaining, removed, pt, due_dates, weights, objective)
            current_val = compute_objectives(current_seq, pt, due_dates, weights)[objective]
            no_improve = 0

        if verbose and (it + 1) % 10 == 0:
            print(f"  Iter {it+1:3d} — best {objective} = {best_val}")

    elapsed = time.time() - start
    final_obj = compute_objectives(best_seq, pt, due_dates, weights)

    if filepath:
        save_detailed_results(
            sequence=best_seq,
            pt=pt,
            due_dates=due_dates,
            weights=weights,
            filepath=filepath,
            extra_summary={
                "Objective optimized": objective,
                "Runtime (s)": elapsed,
                "Iterations": max_iter,
                "k": k,
                "Seed": seed
            }
        )

    if verbose:
        print("Heuristic end — final results:")
        print(f"  TT    = {final_obj['TT']}")
        print(f"  TWT   = {final_obj['TWT']}")
        print(f"  T_max = {final_obj['T_max']}")
        print(f"  NT    = {final_obj['NT']}")
        print(f"  Time  = {elapsed:.2f}s")

    return {
        "sequence": best_seq,
        "TT": final_obj["TT"],
        "TWT": final_obj["TWT"],
        "T_max": final_obj["T_max"],
        "NT": final_obj["NT"],
        "time": elapsed,
        "history": history
    }"""


# ─────────────────────────────────────────────
# TARGETED LOCAL SEARCH
# ─────────────────────────────────────────────
def targeted_jobs(sequence, pt, due_dates, weights, max_jobs=6):
    obj = compute_objectives(sequence, pt, due_dates, weights)
    Tj = obj['Tj']

    scored = []
    for pos, job in enumerate(sequence):
        score = 3.0 * Tj[pos]
        scored.append((score, pos, job))

    scored.sort(reverse=True, key=lambda x: x[0])

    jobs = []
    seen = set()
    for _, _, job in scored:
        if job not in seen:
            jobs.append(job)
            seen.add(job)
        if len(jobs) >= max_jobs:
            break

    return jobs


def local_search_targeted(sequence, pt, due_dates, weights, objective, max_rounds=10):
    current = sequence[:]
    current_val = compute_objectives(current, pt, due_dates, weights)[objective]

    n_jobs = len(sequence)
    max_jobs = n_jobs // 2

    improved = True
    rounds = 0

    while improved and rounds < max_rounds:
        improved = False
        rounds += 1

        candidate_jobs = targeted_jobs(current, pt, due_dates, weights, max_jobs=max_jobs)

        for job in candidate_jobs:
            i = current.index(job)
            partial = current[:i] + current[i+1:]

            best_seq = current
            best_val = current_val

            for j in range(len(partial) + 1):
                if j == i:
                    continue

                cand = partial[:j] + [job] + partial[j:]
                val = compute_objectives(cand, pt, due_dates, weights)[objective]

                if val < best_val:
                    best_val = val
                    best_seq = cand

            if best_val < current_val:
                current = best_seq
                current_val = best_val
                improved = True

    return current



# ─────────────────────────────────────────────
# ELITE POOL
# ─────────────────────────────────────────────
def add_to_elite(elite_pool, sequence, pt, due_dates, weights, objective, max_size=3):
    val = compute_objectives(sequence, pt, due_dates, weights)[objective]

    # évite doublons
    seq_tuple = tuple(sequence)
    for old_seq, old_val in elite_pool:
        if tuple(old_seq) == seq_tuple:
            return elite_pool

    elite_pool.append((sequence[:], val))
    elite_pool.sort(key=lambda x: x[1])

    return elite_pool[:max_size]


def path_relinking(seq_a, seq_b, pt, due_dates, weights, objective):
    current = seq_a[:]
    best = current[:]
    best_val = compute_objectives(best, pt, due_dates, weights)[objective]

    for target_pos in range(len(seq_b)):
        wanted_job = seq_b[target_pos]
        current_pos = current.index(wanted_job)

        if current_pos != target_pos:
            job = current[current_pos]
            partial = current[:current_pos] + current[current_pos+1:]
            current = partial[:target_pos] + [job] + partial[target_pos:]

            val = compute_objectives(current, pt, due_dates, weights)[objective]
            if val < best_val:
                best = current[:]
                best_val = val

    return best


# ─────────────────────────────────────────────
# MAIN HEURISTIC
# ─────────────────────────────────────────────
def heuristic_due_date_pfsp(instance,
                            subdir=None,
                            weights_dir="data/weights",
                            weights=None,
                            objective='TT',
                            d=4,
                            max_iter=10,
                            seed=42,
                            elite_size=3,
                            filepath=None,
                            verbose=True):
    rng = random.Random(seed)
    np.random.seed(seed)

    pt = instance['processing_times']
    due_dates = instance['due_date']
    n_jobs = instance['n_jobs']
    
    
    # WEIGHTS MANAGEMENT
    # ─────────────────────────────────────────────
    if weights is None:

        if subdir is not None:
            weights_path = os.path.join(weights_dir, f"{subdir}_weights.csv")

            if os.path.exists(weights_path):
                weights = load_weights(weights_path)
                if verbose:
                    print(f"Weights chargés depuis {weights_path}", flush=True)
            else:
                weights = generate_weights(n_jobs, seed=seed)
                save_weights(weights, weights_path)
                if verbose:
                    print(f"Weights générés et sauvegardés dans {weights_path}", flush=True)
        else:
            weights = generate_weights(n_jobs, seed=seed)
            if verbose:
                print("Weights générés (pas de subdir fourni)", flush=True)



    start = time.time()

    # 1) initialisation multi-règles
    current_seq, current_obj = multi_start_initial_solution(pt, due_dates, weights, objective)
    current_val = current_obj[objective]

    best_seq = current_seq[:]
    best_val = current_val

    elite_pool = []
    elite_pool = add_to_elite(elite_pool, current_seq, pt, due_dates, weights, objective, max_size=elite_size)

    history = [best_val]

    if verbose:
        print(f"Heuristic start — objective={objective} initial={best_val}")

    no_improve = 0

    for it in range(max_iter):
        # 2) destruction aléatoire
        remaining, removed = destruction_random(current_seq, d, rng=rng)

        # 3) reconstruction gloutonne
        new_seq = reconstruction_greedy(remaining, removed, pt, due_dates, weights, objective)

        # 4) local search ciblée
        new_seq = local_search_targeted(new_seq, pt, due_dates, weights, objective, max_rounds=8)

        # 5) intensification par elite / path relinking de temps en temps
        if elite_pool and (it + 1) % 10 == 0:
            ref_seq = rng.choice(elite_pool)[0]
            pr_seq = path_relinking(new_seq, ref_seq, pt, due_dates, weights, objective)
            pr_val = compute_objectives(pr_seq, pt, due_dates, weights)[objective]
            new_val = compute_objectives(new_seq, pt, due_dates, weights)[objective]

            if pr_val < new_val:
                new_seq = pr_seq

        new_obj = compute_objectives(new_seq, pt, due_dates, weights)
        new_val = new_obj[objective]

        # 6) acceptation
        if new_val < current_val:
            current_seq = new_seq[:]
            current_val = new_val
            no_improve = 0
        """else:
            # acceptation faible pour diversification
            if rng.random() < 0.10:
                current_seq = new_seq[:]
                current_val = new_val
            no_improve += 1"""

        # 7) update best / elite
        if new_val < best_val:
            best_seq = new_seq[:]
            best_val = new_val
            no_improve = 0

        elite_pool = add_to_elite(elite_pool, new_seq, pt, due_dates, weights, objective, max_size=elite_size)
        history.append(best_val)

        # 8) diversification si stagnation
        if no_improve >= 10:
            d_big = min(max(d + 2, 5), len(current_seq) // 2 if len(current_seq) > 2 else 1)
            remaining, removed = destruction_random(current_seq, d_big, rng=rng)
            current_seq = reconstruction_greedy(remaining, removed, pt, due_dates, weights, objective)
            current_val = compute_objectives(current_seq, pt, due_dates, weights)[objective]
            no_improve = 0

        if verbose and (it + 1) % 10 == 0:
            print(f"  Iter {it+1:3d} — best {objective} = {best_val}")

    elapsed = time.time() - start
    final_obj = compute_objectives(best_seq, pt, due_dates, weights)

    if filepath:
        save_detailed_results(
            sequence=best_seq,
            pt=pt,
            due_dates=due_dates,
            weights=weights,
            filepath=filepath,
            extra_summary={
                "Objective optimized": objective,
                "Runtime (s)": elapsed,
                "Iterations": max_iter,
                "d": d,
                "Seed": seed
            }
        )

    if verbose:
        print("Heuristic end — final results:")
        print(f"  TT    = {final_obj['TT']}")
        print(f"  TWT   = {final_obj['TWT']}")
        print(f"  T_max = {final_obj['T_max']}")
        print(f"  NT    = {final_obj['NT']}")
        print(f"  Time  = {elapsed:.2f}s")

    return {
        "sequence": best_seq,
        "TT": final_obj["TT"],
        "TWT": final_obj["TWT"],
        "T_max": final_obj["T_max"],
        "NT": final_obj["NT"],
        "time": elapsed,
        "history": history
    }
