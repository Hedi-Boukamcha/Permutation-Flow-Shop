import csv
import numpy as np
import os

from config import FOLDE_MAP
from src.dd_generator import generate_all_scenarios, generate_due_dates_brah

def parse_taillard(filepath):
    instances = []

    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    i = 0
    while i < len(lines):
        if "number of jobs" in lines[i].lower():
            i += 1
            header = lines[i].split()
            n_jobs     = int(header[0])
            n_machines = int(header[1])
            # header[2] = seed → ignoré
            ub         = int(header[3])
            lb         = int(header[4])

            i += 1  # "processing times :"
            i += 1

            matrix = []
            for m in range(n_machines):
                row = list(map(int, lines[i].split()))
                matrix.append(row)
                i += 1

            instances.append({
                "n_jobs":           n_jobs,
                "n_machines":       n_machines,
                "ub":               ub,
                "lb":               lb,
                "processing_times": np.array(matrix)
            })
        else:
            i += 1

    return instances


def load_instance(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        data = list(reader)

    n_jobs = len(data)
    n_machines = len(header) - 2

    pt = np.zeros((n_machines, n_jobs), dtype=int)
    due_dates = np.zeros(n_jobs, dtype=int)

    for j, row in enumerate(data):
        due_dates[j] = int(row[1])

        for i in range(n_machines):
            pt[i][j] = int(row[2 + i])

    return {
        'processing_times': pt,
        'due_date': due_dates,
        'n_jobs': n_jobs,
        'n_machines': n_machines
    }


def load_all(instances_dir):
    instances = {}
    
    for subdir in ['20j_5m', '50j_10m']:
        instance_dir = os.path.join(instances_dir, subdir)
        instances[subdir] = []
        
        for filename in sorted(os.listdir(instance_dir)):
            if filename.endswith('.csv'):
                instance_path = os.path.join(instance_dir, filename)
                pt, due_dates = load_instance(instance_path)
                
                instance_num = int(filename.split('_')[1].split('.')[0])
                instances[subdir].append({
                    'instance_num': instance_num,
                    'processing_times': pt,
                    'due_dates': due_dates
                })
    
    return instances


def display_dataset(datasets):
    """
    Affiche tous les datasets avec leurs instances et due dates générées.
    """
    T_values = [0.2, 0.4, 0.6, 0.8]
    R_values = [0.2, 0.6, 1.0]

    for name, instances in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset : {name}")
        print(f"Nombre d'instances : {len(instances)}")

        for idx, inst in enumerate(instances):
            print(f"\n  Instance {idx+1}:")
            print(f"    Jobs     : {inst['n_jobs']}")
            print(f"    Machines : {inst['n_machines']}")
            print(f"    UB       : {inst['ub']}")
            print(f"    LB       : {inst['lb']}")
            print(f"    Shape    : {inst['processing_times'].shape}")
            print(f"    PT[0]    : {inst['processing_times'][0]}")

            # Due dates pour chaque scénario (T, R)
            print(f"\n    Due dates générées :")
            scenarios = generate_all_scenarios(inst, seed=42)
            for key, due_dates in scenarios.items():
                print(f"      {key}: min={due_dates.min():4d}, "
                      f"max={due_dates.max():4d}, "
                      f"mean={due_dates.mean():7.1f}")
                

def save_instances(datasets, output_dir="data/instances"):
    """
    Sauvegarde chaque instance dans un fichier CSV.
    Une ligne par job avec toutes les due dates en colonnes.
    """

    for name, instances in datasets.items():
        folder_name = FOLDE_MAP.get(name, name)
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for idx, instance in enumerate(instances):
            filepath = os.path.join(folder_path, f"instance_{idx+1}.csv")

            with open(filepath, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                n_machines = instance['n_machines']
                n_jobs     = instance['n_jobs']
                pt         = instance['processing_times']

                # Due dates avec tau=2
                due_dates = generate_due_dates_brah(instance, tau=2)
                                
                # Ajouter les dates dues à l'instance
                instance['due_dates'] = due_dates

                # En-tête
                machine_cols = [f"p_machine{i+1}" for i in range(n_machines)]
                writer.writerow(["job", "due_date"] + machine_cols)

                # Une ligne par job
                for job_idx in range(n_jobs):
                    job_pt = [int(pt[m][job_idx]) for m in range(n_machines)]
                    writer.writerow([
                        job_idx + 1,
                        int(due_dates[job_idx])
                    ] + job_pt)

