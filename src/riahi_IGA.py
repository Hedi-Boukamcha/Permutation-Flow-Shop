import random
import numpy as np
import time

from src.dd_generator import generate_due_dates_brah
from src.scheduler import compute_completion_times


def compute_tt_fast(sequence, pt, due_dates):
    """
    Calcule la Tardivité Totale (TT) de manière efficace.
    
    Args:
        sequence (list/array): Séquence des jobs (indices 0-based).
        pt (ndarray): Matrice des temps de traitement (machines x jobs).
        due_dates (ndarray): Vecteur des dates d'échéance indexé par job.
        
    Returns:
        int: Somme des tardivités max(0, C_j - d_j).
    """
    # 1. Calcul des dates de fin (C_ij) via ton scheduler
    C = compute_completion_times(sequence, pt)
    
    # 2. Extraction des dates de fin sur la dernière machine (Cm)
    # C[-1] contient les temps de fin de chaque job dans l'ordre de la séquence
    completion_times_last_machine = C[-1]
    
    # 3. Récupération des due dates correspondant aux jobs dans l'ordre de la séquence
    # On convertit en liste si c'est un ndarray pour l'indexation
    ordered_due_dates = due_dates[list(sequence)]
    
    # 4. Calcul de la tardivité : max(0, C - d)
    # np.maximum est vectorisé et beaucoup plus rapide qu'une boucle Python
    tardiness = np.maximum(completion_times_last_machine - ordered_due_dates, 0)
    
    return int(np.sum(tardiness))


def iga_riahi_final(instance, due_dates, max_time=5.0):
    """
    Interface pour l'algorithme Iterated Greedy (IGA).
    Utilise les paramètres optimaux de Riahi et al. (2020) :
    d=4, beta=0.5, T=0.4
    """
    start_cpu = time.perf_counter()
    
    # Appel de la logique de recherche itérative
    # On passe les hyperparamètres optimaux trouvés dans l'article
    best_seq, best_tt = iga_best_config(
        instance, 
        due_dates, 
        max_time=max_time
    )
    
    elapsed = time.perf_counter() - start_cpu
    
    # On retourne 0 pour 'ties' car l'IGA n'est pas une heuristique 
    # de construction simple comme le NEH
    return best_seq, 0, elapsed


def iga_best_config(instance, due_dates, max_time=5.0):
    """
    Configuration optimisée selon Riahi et al. (2020)
    Paramètres : d=4, beta=0.5, T=0.4
    """
    # --- Hyperparamètres fixés sur les meilleurs résultats de l'article ---
    D_SIZE = 4
    BETA = 0.5
    T_CONST = 0.4

    pt = instance['processing_times']
    n_jobs = instance['n_jobs']
    n_m = instance['n_machines']

    # 1. Initialisation : NEHedd (déjà implémenté dans ton code)
    # Ici on simule par un tri EDD suivi d'une recherche locale
    current_seq = np.argsort(due_dates).tolist()
    current_tt = compute_tt_fast(current_seq, pt, due_dates)
    current_seq, current_tt = local_search_riahi(current_seq, pt, due_dates, BETA)
    
    best_seq = list(current_seq)
    best_tt = current_tt

    # Température calibrée : T = 0.4 * (Somme p_ij / (n * m * 10))
    temp = T_CONST * (np.sum(pt) / (n_jobs * n_m * 10))
    
    start_time = time.perf_counter()

    while (time.perf_counter() - start_time) < max_time:
        # --- DESTRUCTION ---
        working_seq = list(current_seq)
        removed = []
        # Extraction de d=4 jobs au hasard
        indices = random.sample(range(len(working_seq)), D_SIZE)
        indices.sort(reverse=True)
        for i in indices:
            removed.append(working_seq.pop(i))

        # --- CONSTRUCTION ---
        # Réinsertion type NEH (meilleure position pour chaque job retiré)
        for job in removed:
            best_p = -1
            min_tt_found = float('inf')
            for pos in range(len(working_seq) + 1):
                candidate = working_seq[:pos] + [job] + working_seq[pos:]
                tt = compute_tt_fast(candidate, pt, due_dates)
                if tt < min_tt_found:
                    min_tt_found = tt
                    best_p = pos
            working_seq.insert(best_p, job)
        
        # --- RECHERCHE LOCALE (Basée sur BETA) ---
        # L'article utilise beta pour décider si on applique la recherche locale 
        # sur la solution reconstruite pour intensifier la recherche.
        if random.random() < BETA:
            candidate_seq, candidate_tt = local_search_riahi(working_seq, pt, due_dates, BETA)
        else:
            candidate_seq = working_seq
            candidate_tt = compute_tt_fast(candidate_seq, pt, due_dates)

        # --- ACCEPTATION (METROPOLIS) ---
        if candidate_tt < current_tt:
            current_tt = candidate_tt
            current_seq = list(candidate_seq)
            if current_tt < best_tt:
                best_tt = current_tt
                best_seq = list(current_seq)
        elif random.random() <= np.exp(-(candidate_tt - current_tt) / temp):
            current_tt = candidate_tt
            current_seq = list(candidate_seq)

    return best_seq, best_tt

def local_search_riahi(sequence, pt, due_dates, beta):
    """
    Recherche locale par insertion (SMI)
    """
    best_s = list(sequence)
    best_t = compute_tt_fast(best_s, pt, due_dates)
    
    # On parcourt tous les jobs dans un ordre aléatoire
    idxs = list(range(len(best_s)))
    random.shuffle(idxs)
    
    for i in idxs:
        job = best_s.pop(i)
        target_pos = 0
        min_tt = float('inf')
        
        for p in range(len(best_s) + 1):
            temp_s = best_s[:p] + [job] + best_s[p:]
            tt = compute_tt_fast(temp_s, pt, due_dates)
            if tt < min_tt:
                min_tt = tt
                target_pos = p
        
        best_s.insert(target_pos, job)
        if min_tt < best_t:
            best_t = min_tt
            
    return best_s, best_t

def run_on_instances(name, instances, algo_func):
    results = []
    print(f"\n  Dataset : {name} ({len(instances)} instances)")

    for idx, inst in enumerate(instances):
        # Génération des due dates (même méthode pour tous)
        due_dates = generate_due_dates_brah(inst, tau=2)
        pt = inst['processing_times']

        # APPEL DYNAMIQUE : algo_func est soit nehedd_tbit1, soit iga_riahi_final
        # Les deux doivent retourner (sequence, ties, elapsed)
        sequence, ties, elapsed = algo_func(inst, due_dates)

        # Calcul du TT final (toujours avec la même méthode pour comparer)
        tt = compute_tt_fast(sequence, pt, due_dates)

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
        print(f"    Inst {idx+1:2d} | TT={tt:8d} | CPU={elapsed:.4f}s")

    return results