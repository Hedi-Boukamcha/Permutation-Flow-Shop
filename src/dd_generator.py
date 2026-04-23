import csv
import os
import numpy as np

from src.data_loader import load_instance


def generate_due_dates(instance, T, R, seed=None):
    """
    Génère les due dates selon Potts & Van Wassenhove.
    
    Args:
        instance : dict avec 'lb', 'n_jobs', 'processing_times'
        T : tardiness factor (ex: 0.2, 0.4, 0.6, 0.8)
        R : due date range (ex: 0.2, 0.6, 1.0)
        seed : pour reproductibilit
    
    Returns:
        due_dates : array de taille n_jobs
    """
    if seed is not None:
        np.random.seed(seed)
    
    P = instance['lb']  # borne inférieure du makespan
    n = instance['n_jobs']
    
    lower = P * (1 - T - R/2)
    upper = P * (1 - T + R/2)
    
    # S'assurer que les bornes sont positives
    lower = max(1, lower)
    upper = max(lower + 1, upper)
    
    due_dates = np.random.uniform(lower, upper, size=n)
    due_dates = np.round(due_dates).astype(int)
    
    return due_dates


def generate_all_scenarios(instance, seed=42):
    """
    Génère tous les scénarios (T, R) pour une instance.
    Retourne un dict avec les due dates pour chaque scénario.
    """
    T_values = [0.2, 0.4, 0.6, 0.8]
    R_values = [0.2, 0.6, 1.0]
    
    scenarios = {}
    for T in T_values:
        for R in R_values:
            key = f"T{T}_R{R}"
            scenarios[key] = generate_due_dates(instance, T, R, seed=seed)
    return scenarios


def generate_due_dates_brah(instance, tau=2):
    pt = instance['processing_times']  # (n_machines, n_jobs)
    
    # Somme des temps de traitement par job
    total_work = pt.sum(axis=0)  # shape: (n_jobs,)
    
    due_dates = tau * total_work
    
    return due_dates.astype(int)


def generate_weights(instance, seed=42):
    """
    Génère les poids w_j pour chaque job.
    w_j ~ U[1, 10] (distribution uniforme entière)

    Args:
        instance : dict avec 'n_jobs'
        seed     : pour reproductibilité

    Returns:
        weights : array de taille n_jobs
    """
    np.random.seed(seed)
    n_jobs  = instance['n_jobs']
    weights = np.random.randint(1, 11, size=n_jobs)
    return weights

def save_weights_per_size(instances_dir, output_dir="data/weights", seed=42):

    for subdir in os.listdir(instances_dir):

        subdir_path = os.path.join(instances_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # 👉 récupérer UNE instance pour connaître n_jobs
        instance_file = sorted([
            f for f in os.listdir(subdir_path) if f.endswith(".csv")
        ])[0]

        instance_path = os.path.join(subdir_path, instance_file)
        instance = load_instance(instance_path)

        n_jobs = instance['n_jobs']

        weights = generate_weights(n_jobs, seed=seed)

        # 👉 fichier unique par taille
        output_path = os.path.join(output_dir, f"{subdir}_weights.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["job", "weight"])

            for j, w in enumerate(weights):
                writer.writerow([j+1, int(w)])

        print(f"Weights sauvegardés : {output_path}")


def load_weights_per_size(subdir, weights_dir="data/weights"):
    filepath = os.path.join(weights_dir, f"{subdir}_weights.csv")

    weights = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            weights.append(int(row[1]))

    return np.array(weights)