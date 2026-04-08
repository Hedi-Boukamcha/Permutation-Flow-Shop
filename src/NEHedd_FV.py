"""
NEHedd avec mécanisme de tie-breaking IT1 (Fernandez-Viagas & Framinan, 2015).
Objectif unique : minimisation de la tardivité totale (TT).

IT1 = Σᵢ (C_{i,n} − Σⱼ p_{i,j}) — idle time total, front delays inclus.
Minimiser IT1 sert uniquement de tie-breaker quand plusieurs positions donnent le même TT.

Dépendances :
    src/scheduler.py    → compute_completion_times()
    src/dd_generator.py → generate_due_dates_brah()
    src/data_loader.py  → load_all()

Résultats :
    resultats/20j_5m_results.csv
    resultats/50j_10m_results.csv
"""

import os
import csv
import time
import numpy as np

from src.scheduler    import compute_completion_times
from src.dd_generator import generate_due_dates_brah
from src.data_loader  import load_all


# ─── Calcul du TT à partir des completion times ──────────────────────────────────

def compute_tt(sequence, pt, due_dates):
    """
    TT = Σⱼ max(C_{m,j} − d_j, 0)

    Utilise compute_completion_times() de scheduler.py.
    """
    C  = compute_completion_times(sequence, pt)   # (n_machines, n_jobs)
    Cm = C[-1]                                    # completion times sur dernière machine
    dj = due_dates[list(sequence)]
    return int(np.maximum(Cm - dj, 0).sum())


# ─── Calcul de IT1 (tie-breaker) ─────────────────────────────────────────────────

def compute_it1(sequence, pt):
    """
    IT1 = Σᵢ (C_{i,n} − Σⱼ p_{i,j})

    Utilise compute_completion_times() de scheduler.py.
    Minimiser IT1 ≡ minimiser la somme des completion times sur toutes les machines.
    """
    C   = compute_completion_times(sequence, pt)
    it1 = 0
    for i in range(pt.shape[0]):
        sum_p = int(sum(pt[i][job] for job in sequence))
        it1  += int(C[i][-1]) - sum_p
    return it1


# ─── Évaluation d'une insertion ──────────────────────────────────────────────────

def evaluate_insertion(partial_seq, job, pt, due_dates):
    """
    Teste l'insertion de `job` dans chaque position de `partial_seq`.

    Critère primaire : TT minimal.
    Tie-breaking     : IT1 minimal en cas d'égalité de TT.

    Returns:
        best_pos   : position d'insertion optimale
        ties_count : nombre de positions avec le même TT minimal (avant tie-break)
    """
    best_tt    = None
    best_it1   = None
    best_pos   = 0
    ties_count = 0

    for pos in range(len(partial_seq) + 1):
        candidate = list(partial_seq[:pos]) + [job] + list(partial_seq[pos:])
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


# ─── NEHedd avec TBIT1 ───────────────────────────────────────────────────────────

def nehedd_tbit1(instance, due_dates):
    """
    Algorithme NEHedd avec tie-breaking IT1.

    Ordre initial : EDD — tri par due date croissante (stable).
    Critère       : TT minimal à chaque insertion.
    Tie-breaking  : IT1 minimal en cas d'égalité.

    Args:
        instance  : dict avec 'processing_times', 'n_jobs'
        due_dates : np.array (n_jobs,)

    Returns:
        sequence   : liste d'indices de jobs 0-based (solution finale)
        total_ties : nombre total de tie-breaks effectués
        elapsed    : temps CPU en secondes
    """
    pt     = instance['processing_times']
    n_jobs = instance['n_jobs']

    t_start = time.perf_counter()

    # Étape 1 : ordre EDD
    edd_order = np.argsort(due_dates, kind='stable').tolist()

    # Étape 2 : construction itérative
    sequence   = [edd_order[0]]
    total_ties = 0

    for k in range(1, n_jobs):
        job = edd_order[k]
        best_pos, ties = evaluate_insertion(sequence, job, pt, due_dates)
        total_ties += max(0, ties - 1)
        sequence.insert(best_pos, job)

    elapsed = time.perf_counter() - t_start
    return sequence, total_ties, elapsed


# ─── Sauvegarde CSV ──────────────────────────────────────────────────────────────

FIELDNAMES = [
    "instance",
    "n_jobs",
    "n_machines",
    "lb",
    "ub",
    "TT",
    "total_ties",
    "cpu_time_s",
    "sequence",
]


def save_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Résultats sauvegardés → {filepath}")


# ─── Pipeline principal ──────────────────────────────────────────────────────────

def run_dataset(name, instances):
    results = []
    print(f"\n  Dataset : {name} ({len(instances)} instances)")

    for idx, inst in enumerate(instances):
        pt        = inst['processing_times']
        due_dates = generate_due_dates_brah(inst, tau=2)

        sequence, ties, elapsed = nehedd_tbit1(inst, due_dates)
        tt = compute_tt(sequence, pt, due_dates)

        row = {
            "instance":   idx + 1,
            "n_jobs":     inst['n_jobs'],
            "n_machines": inst['n_machines'],
            "lb":         inst['lb'],
            "ub":         inst['ub'],
            "TT":         tt,
            "total_ties": ties,
            "cpu_time_s": round(elapsed, 6),
            "sequence":   " ".join(str(j + 1) for j in sequence),
        }
        results.append(row)

        print(f"    Instance {idx+1:2d} | TT={tt:8d} | Ties={ties:4d} | CPU={elapsed:.4f}s")

    return results


def main(data_dir="data/taillard", output_dir="resultats"):
    print("=" * 55)
    print("  NEHedd — Tie-Breaking IT1 (Fernandez-Viagas 2015)")
    print("  Objectif : minimisation TT")
    print("=" * 55)

    datasets = load_all(data_dir)

    folder_map = {
        "tai20j_5m":  "20j_5m",
        "tai50j_10m": "50j_10m",
    }

    for name, instances in datasets.items():
        results  = run_dataset(name, instances)
        label    = folder_map.get(name, name)
        filepath = os.path.join(output_dir, f"{label}_results.csv")
        save_results(results, filepath)

    print("\nTerminé.")




if __name__ == "__main__":
    main()