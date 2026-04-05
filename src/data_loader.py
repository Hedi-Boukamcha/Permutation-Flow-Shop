import numpy as np
import os

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
            for _ in range(n_machines):
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


def load_all(data_dir="data/raw"):
    datasets = {}
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            name = filename.replace(".txt", "")
            filepath = os.path.join(data_dir, filename)
            datasets[name] = parse_taillard(filepath)
            print(f"Chargé : {name} — {len(datasets[name])} instances")
    return datasets