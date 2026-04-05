import csv
import os

from src.dd_generator import generate_all_scenarios
from src.data_loader import load_all


def save_to_csv(datasets, output_dir="data/taillard"):
    """
    Sauvegarde tous les datasets avec due dates et processing times dans des fichiers CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, instances in datasets.items():
        filepath = os.path.join(output_dir, f"{name}.csv")

        with open(filepath, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Récupérer n_machines pour créer les colonnes machine
            n_machines = instances[0]['n_machines']
            machine_cols = [f"p_machine{i+1}" for i in range(n_machines)]

            # En-tête
            writer.writerow([
                "instance", 
                #"n_jobs", 
                # "n_machines", 
                "ub",
                "lb",
                "T", 
                "R", 
                "job",
                "due_date"
            ] + machine_cols)

            for idx, inst in enumerate(instances):
                scenarios = generate_all_scenarios(inst, seed=42)
                pt = inst['processing_times']  # shape (n_machines, n_jobs)

                for key, due_dates in scenarios.items():
                    parts = key.split("_")
                    T = parts[0][1:]
                    R = parts[1][1:]

                    for job_idx, due_date in enumerate(due_dates):
                        # Processing times de ce job sur chaque machine
                        job_pt = [pt[m][job_idx] for m in range(n_machines)]

                        writer.writerow([
                            idx + 1,
                            #inst['n_jobs'],
                            #inst['n_machines'],
                            inst['ub'],
                            inst['lb'],
                            T,
                            R,
                            job_idx + 1,
                            due_date
                        ] + job_pt)

        print(f"Fichier sauvegardé : {filepath}")

