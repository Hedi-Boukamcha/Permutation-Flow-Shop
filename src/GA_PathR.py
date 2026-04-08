# src/ga_pr.py

import numpy as np
import random
import time
from src.scheduler import compute_objectives
from src.initial_solution import nehedd
from src.dd_generator import generate_due_dates_brah, generate_weights
from config import FOLDE_MAP
from src.results import save_results


# ─────────────────────────────────────────────────────────
# OPÉRATEURS GÉNÉTIQUES
# ─────────────────────────────────────────────────────────

def ox_crossover(parent1, parent2):
    """
    Order Crossover (OX) — Vallada & Ruiz (2010)
    Préserve l'ordre relatif des jobs.

    Args:
        parent1, parent2 : séquences parents

    Returns:
        child1, child2 : séquences enfants
    """
    n = len(parent1)

    # Choisir deux points de coupe
    cut1, cut2 = sorted(random.sample(range(n), 2))

    def _ox(p1, p2):
        # Segment central de p1
        segment = p1[cut1:cut2]
        # Compléter avec les jobs de p2 dans l'ordre
        remaining = [j for j in p2 if j not in segment]
        child = remaining[:cut1] + segment + remaining[cut1:]
        return child

    child1 = _ox(parent1, parent2)
    child2 = _ox(parent2, parent1)

    return child1, child2


def mutation_insertion(sequence):
    """
    Mutation par insertion — retirer un job et le réinsérer ailleurs.

    Args:
        sequence : séquence à muter

    Returns:
        mutated : séquence mutée
    """
    n   = len(sequence)
    seq = sequence[:]

    # Choisir un job aléatoire
    i = random.randint(0, n - 1)
    job = seq.pop(i)

    # Réinsérer à une position aléatoire différente
    j = random.randint(0, n - 1)
    seq.insert(j, job)

    return seq


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Sélection par tournoi.

    Args:
        population     : liste de séquences
        fitnesses      : liste de valeurs TT correspondantes
        tournament_size: nombre de candidats dans le tournoi

    Returns:
        winner : séquence gagnante
    """
    candidates = random.sample(range(len(population)), tournament_size)
    winner     = min(candidates, key=lambda i: fitnesses[i])
    return population[winner][:]


# ─────────────────────────────────────────────────────────
# PATH RELINKING
# ─────────────────────────────────────────────────────────

def path_relinking(source, target, processing_times, due_dates,
                   weights, objective):
    """
    Path Relinking entre source et target.
    Génère des solutions intermédiaires en transformant source → target.
    Retourne la meilleure solution trouvée.

    Args:
        source, target   : séquences de départ et d'arrivée
        processing_times : matrice (n_machines x n_jobs)
        due_dates        : array (n_jobs,)
        weights          : array (n_jobs,)
        objective        : 'TT'

    Returns:
        best_seq   : meilleure séquence trouvée
        best_value : valeur objectif
    """
    current    = source[:]
    best_seq   = source[:]
    obj        = compute_objectives(source, processing_times, due_dates, weights)
    best_value = obj[objective]

    for i in range(len(target)):
        if current[i] == target[i]:
            continue

        # Trouver la position du job target[i] dans current
        j = current.index(target[i])

        # Déplacer target[i] à la position i (insert move)
        job = current.pop(j)
        current.insert(i, job)

        # Évaluer la solution intermédiaire
        obj   = compute_objectives(current, processing_times, due_dates, weights)
        value = obj[objective]

        if value < best_value:
            best_value = value
            best_seq   = current[:]

    return best_seq, best_value


# ─────────────────────────────────────────────────────────
# ALGORITHME PRINCIPAL GA + PR
# ─────────────────────────────────────────────────────────

def ga_pr(processing_times, due_dates, weights=None, objective='TT',
          pop_size=10, max_time=None, n_jobs_param=None, n_machines_param=None,
          seed=42):
    """
    Genetic Algorithm with Path Relinking — Vallada & Ruiz (2010)
    Omega, 38(1-2), 57-67.

    Paramètres calibrés selon l'article :
    - pop_size      = 10
    - Croisement    = OX
    - Mutation      = insertion
    - Path Relinking = entre meilleur et pire
    - Critère arrêt = n × m × t secondes (t calibré dans l'article)

    Args:
        processing_times  : matrice (n_machines x n_jobs)
        due_dates         : array (n_jobs,)
        weights           : array (n_jobs,)
        objective         : 'TT'
        pop_size          : taille population (défaut=10)
        max_time          : temps max en secondes
        n_jobs_param      : n_jobs (pour critère arrêt)
        n_machines_param  : n_machines (pour critère arrêt)
        seed              : reproductibilité

    Returns:
        best_sequence : meilleure séquence
        best_value    : valeur objectif
        history       : historique
    """
    random.seed(seed)
    np.random.seed(seed)

    n_machines = processing_times.shape[0]
    n_jobs     = processing_times.shape[1]

    # Critère d'arrêt — n × m × 0.4 secondes (meilleur selon l'article)
    if max_time is None:
        max_time = n_jobs * n_machines * 0.4

    start_time = time.time()

    # ── 1. Initialisation population ─────────────────────
    population = []
    fitnesses  = []

    # Premier individu → NEHedd (meilleure solution initiale)
    seq_neh = nehedd(processing_times, due_dates, weights, objective=objective)
    population.append(seq_neh)
    obj = compute_objectives(seq_neh, processing_times, due_dates, weights)
    fitnesses.append(obj[objective])

    # Reste de la population → aléatoire
    for _ in range(pop_size - 1):
        seq = list(range(n_jobs))
        random.shuffle(seq)
        population.append(seq)
        obj = compute_objectives(seq, processing_times, due_dates, weights)
        fitnesses.append(obj[objective])

    # Meilleure solution initiale
    best_idx      = np.argmin(fitnesses)
    best_sequence = population[best_idx][:]
    best_value    = fitnesses[best_idx]
    history       = [best_value]

    print(f"  GA-PR démarrage — {objective} initial : {best_value}")

    # ── 2. Boucle principale ──────────────────────────────
    generation = 0

    while time.time() - start_time < max_time:
        generation += 1

        # a. Sélection par tournoi
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)

        # b. Croisement OX
        child1, child2 = ox_crossover(parent1, parent2)

        # c. Mutation par insertion
        if random.random() < 0.1:  # taux mutation 10%
            child1 = mutation_insertion(child1)
        if random.random() < 0.1:
            child2 = mutation_insertion(child2)

        # d. Évaluer les enfants
        for child in [child1, child2]:
            obj_child = compute_objectives(
                child, processing_times, due_dates, weights
            )
            val_child = obj_child[objective]

            # e. Path Relinking — entre child et meilleure solution
            pr_seq, pr_val = path_relinking(
                source           = child,
                target           = best_sequence,
                processing_times = processing_times,
                due_dates        = due_dates,
                weights          = weights,
                objective        = objective
            )

            # Garder le meilleur entre child et PR
            if pr_val < val_child:
                final_seq = pr_seq
                final_val = pr_val
            else:
                final_seq = child
                final_val = val_child

            # f. Remplacement — remplacer le pire individu
            worst_idx = np.argmax(fitnesses)
            if final_val < fitnesses[worst_idx]:
                population[worst_idx] = final_seq[:]
                fitnesses[worst_idx]  = final_val

            # g. Mettre à jour meilleure solution globale
            if final_val < best_value:
                best_sequence = final_seq[:]
                best_value    = final_val

        history.append(best_value)

    elapsed = round(time.time() - start_time, 4)
    print(f"  GA-PR terminé — {objective} final : {best_value} "
          f"({generation} générations, {elapsed}s)")

    return best_sequence, best_value, history


def ga_pr_wrapper(inst, due_dates):
    """
    Wrapper pour GA-PR compatible avec run_riahi_IGA.
    Retourne (sequence, ties, elapsed).
    """
    import time

    pt      = inst['processing_times']
    weights = generate_weights(inst)

    start = time.time()

    sequence, _, _ = ga_pr(
        processing_times = pt,
        due_dates        = due_dates,
        weights          = weights,
        objective        = 'TT',
        pop_size         = 10,
        seed             = 42
    )

    elapsed = time.time() - start
    ties    = 0  # non utilisé

    return sequence, ties, elapsed