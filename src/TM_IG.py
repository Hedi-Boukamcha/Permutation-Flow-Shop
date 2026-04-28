"""
TM-IG : Tabu Memory based Iterated Greedy
==========================================
Implémentation basée sur :
    Feng, Zhao, Jiang, Tao, Mei (2023)
    "A tabu memory based iterated greedy algorithm for the distributed
     heterogeneous permutation flowshop scheduling problem with the
     total tardiness criterion"
    Expert Systems with Applications

Compatible avec run_riahi_IGA() — le wrapper retourne (sequence, ties, elapsed)
comme nehedd_tbit1 et iga_riahi_final.

Paramètres retenus (meilleurs résultats selon l'article) :
    d         = 4        (jobs détruits par itération)
    T_factor  = 0.4      (température = 0.4 × ΣP / (n × m))
    tabu_size = n // 2   (taille liste tabou)
"""

import numpy as np
import random
import time

from src.initial_solution import nehedd
from src.results import save_results
from src.scheduler import compute_completion_times, compute_objectives
from src.dd_generator import generate_due_dates_brah, generate_weights


# ---------------------------------------------------------------------------
# Calcul du TT (même style que compute_tt_fast dans riahi_IGA.py)
# ---------------------------------------------------------------------------

def _compute_tt(sequence, pt, due_dates):
    C  = compute_completion_times(sequence, pt)
    Cj = C[-1]
    dj = due_dates[list(sequence)]
    return int(np.maximum(Cj - dj, 0).sum())


# ---------------------------------------------------------------------------
# Initialisation : NEHedd (tri EDD + insertion greedy)
# ---------------------------------------------------------------------------

def _nehedd_init(pt, due_dates, weights=None, objective="TT"):
    """NEHedd simple — tri EDD puis insertion meilleure position."""
    n_jobs    = pt.shape[1]
    edd_order = np.argsort(due_dates, kind='stable').tolist()

    sequence = [edd_order[0]]
    for k in range(1, n_jobs):
        job      = edd_order[k]
        best_tt  = float('inf')
        best_pos = 0
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            tt = _compute_tt(candidate, pt, due_dates)
            if tt < best_tt:
                best_tt  = tt
                best_pos = pos
        sequence.insert(best_pos, job)

    return sequence


# ---------------------------------------------------------------------------
# Recherche locale par insertion (best improvement)
# ---------------------------------------------------------------------------

def _local_search(sequence, pt, due_dates, weights=None, objective="TT"):
    """
    Local search par insertion (best improvement).
    Parcourt tous les jobs, retire chacun et le réinsère à la meilleure position.
    Répète jusqu'à convergence.
    """
    best_seq = list(sequence)
    best_tt  = _compute_tt(best_seq, pt, due_dates)
    improved = True

    while improved:
        improved = False
        for i in range(len(best_seq)):
            job     = best_seq[i]
            current = best_seq[:i] + best_seq[i+1:]

            local_best_tt  = float('inf')
            local_best_pos = 0
            for pos in range(len(current) + 1):
                candidate = current[:pos] + [job] + current[pos:]
                tt = compute_objectives(candidate, pt, due_dates, weights)[objective]
                if tt < local_best_tt:
                    local_best_tt  = tt
                    local_best_pos = pos

            new_seq = current[:local_best_pos] + [job] + current[local_best_pos:]
            if local_best_tt < best_tt:
                best_tt  = local_best_tt
                best_seq = new_seq
                improved = True
                break  # redémarre depuis le début

    return best_seq, best_tt


# ---------------------------------------------------------------------------
# Destruction
# ---------------------------------------------------------------------------

def _destruction(sequence, d):
    """Retire d jobs au hasard. Retourne (remaining, removed)."""
    indices   = random.sample(range(len(sequence)), d)
    removed   = [sequence[i] for i in sorted(indices)]
    remaining = [j for i, j in enumerate(sequence) if i not in indices]
    return remaining, removed


# ---------------------------------------------------------------------------
# Construction (greedy réinsertion)
# ---------------------------------------------------------------------------

def _construction(remaining, removed, pt, due_dates, weights=None, objective="TT"):
    """Réinsère les jobs retirés un par un à la meilleure position."""
    seq = list(remaining)
    for job in removed:
        best_tt  = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            tt = _compute_tt(candidate, pt, due_dates)
            if tt < best_tt:
                best_tt  = tt
                best_pos = pos
        seq.insert(best_pos, job)
    return seq


# ---------------------------------------------------------------------------
# TM-IG principal
# ---------------------------------------------------------------------------

def tmig(instance, due_dates, weights=None, objective="TT", max_time=5.0, d=5, T_factor=0.4, tabu_size=None, seed=42):
    """
    Tabu Memory based Iterated Greedy (TM-IG).

    Args:
        instance  : dict avec 'processing_times', 'n_jobs', 'n_machines'
        due_dates : array (n_jobs,)
        max_time  : budget temps en secondes
        d         : jobs détruits par itération (recommandé : 4)
        T_factor  : température = T_factor x ΣP / (n x m) (recommandé : 0.4)
        tabu_size : longueur liste tabou (défaut : n // 2)
        seed      : reproductibilité

    Returns:
        best_seq : liste des jobs (solution finale)
        best_tt  : TT de la meilleure solution
    """
    random.seed(seed)
    np.random.seed(seed)

    pt         = instance['processing_times']
    n_jobs     = instance['n_jobs']
    n_machines = instance['n_machines']

    if weights is None:
        weights = np.ones(n_jobs, dtype=int)
    else:
        weights = np.array(weights, dtype=int)

    if len(weights) != n_jobs:
        raise ValueError(
            f"Erreur weights: len(weights)={len(weights)} alors que n_jobs={n_jobs}"
        )

    if tabu_size is None:
        tabu_size = max(1, n_jobs // 2)

    # Température : T = T_factor x (Σ p_ij) / (n x m)
    temperature = T_factor * pt.sum() / (n_jobs * n_machines)

    # ── Initialisation : NEHedd ──────────────────────────
    current_seq = nehedd(
        processing_times=pt,
        due_dates=due_dates,
        weights=weights,
        objective=objective
    )
    current_seq, current_tt = _local_search(current_seq, pt, due_dates, weights, objective)
    best_seq = list(current_seq)
    best_tt  = current_tt

    # Liste tabou : mémorise les TT des solutions visitées
    tabu_list = []

    start_time = time.perf_counter()

    # ── Boucle principale ────────────────────────────────
    while (time.perf_counter() - start_time) < max_time:

        # 1. Destruction
        remaining, removed = _destruction(current_seq, d)

        # 2. Construction
        new_seq = _construction(remaining, removed, pt, due_dates, weights, objective)
        
        new_seq, new_tt = _local_search(new_seq, pt, due_dates, weights, objective)
        # 3. Local search
        #new_seq, new_tt = _local_search(new_seq, pt, due_dates)

        # 4. Aspiration : accepté même si tabou si c'est le meilleur global
        if new_tt < best_tt:
            best_seq    = list(new_seq)
            best_tt     = new_tt
            current_seq = list(new_seq)
            current_tt  = new_tt
            tabu_list.append(new_tt)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
            continue

        # 5. Vérification tabou
        is_tabu = new_tt in tabu_list

        # 6. Acceptation Metropolis (si non tabou)
        if not is_tabu:
            delta = new_tt - current_tt
            if delta <= 0 or random.random() < np.exp(-delta / temperature):
                current_seq = list(new_seq)
                current_tt  = new_tt
                tabu_list.append(new_tt)
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)

    return best_seq, best_tt


# ---------------------------------------------------------------------------
# Wrapper compatible run_riahi_IGA
# Signature : algo_func(inst, due_dates) -> (sequence, ties, elapsed)
# ---------------------------------------------------------------------------

def tmig_wrapper(inst, due_dates):
    """
    Wrapper TM-IG compatible avec run_riahi_IGA().

    Paramètres calés sur les meilleurs résultats de Feng et al. (2023) :
        d         = 4
        T_factor  = 0.4
        tabu_size = n // 2
        max_time  = n x m x 0.4 s  (même critère que GA-PR)

    Returns:
        sequence : liste des jobs (0-based)
        ties     : 0 (non applicable pour TM-IG)
        elapsed  : temps CPU en secondes
    """
    n_jobs     = inst['n_jobs']
    n_machines = inst['n_machines']

    # Critère d'arrêt : n x m x 0.4 s (cohérent avec GA-PR du projet)
    max_time = n_jobs * n_machines * 0.4

    start = time.perf_counter()

    sequence, _ = tmig(
        instance  = inst,
        due_dates = due_dates,
        max_time  = max_time,
        d         = 5,
        T_factor  = 0.4,
        tabu_size = n_jobs // 2,
        seed      = 42,
    )

    elapsed = time.perf_counter() - start

    return sequence, 0, elapsed


def run_tmig(instance, weights=None, objective="TT", filepath=None):
    start = time.time()

    pt = instance["processing_times"]
    due_dates = instance["due_date"]
    n_jobs = instance["n_jobs"]

    if weights is None:
        weights = np.ones(n_jobs, dtype=int)
    else:
        weights = np.array(weights, dtype=int)

    sequence, best_val = tmig(
        instance=instance,
        due_dates=due_dates,
        weights=weights,
        objective=objective,
        max_time=n_jobs * instance["n_machines"] * 0.4,
        d=4,
        T_factor=0.4,
        tabu_size=n_jobs // 2,
        seed=42
    )

    elapsed = time.time() - start

    obj = compute_objectives(
        sequence=sequence,
        processing_times=pt,
        due_dates=due_dates,
        weights=weights
    )

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
        "time": elapsed
    }