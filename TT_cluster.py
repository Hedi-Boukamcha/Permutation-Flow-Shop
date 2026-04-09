import sys
import os
import csv
import numpy as np
from src.data_loader import load_all
from src.dd_generator import generate_due_dates_brah
from src.math_model import solve_milp_tt

if __name__ == "__main__":

    # Récupérer l'index de l'instance depuis SLURM
    instance_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    datasets  = load_all("data/taillard")
    instances = datasets["tai20j_5m"]
    inst      = instances[instance_id]

    for idx, inst in enumerate(instances):
        pt        = inst['processing_times']
        due_dates = generate_due_dates_brah(inst, tau=2)

        print(f"\n{'='*50}")
        print(f"Instance {idx+1} — tai20j_5m")

        result = solve_milp_tt(
            processing_times = pt,
            due_dates        = due_dates,
            time_limit       = 18000,        # 5 heures = 5 × 3600
        )
        
    # ── Sauvegarde ───────────────────────────────────────
        # Sauvegarde individuelle
    os.makedirs("resultats/milp_TT/20j_5m", exist_ok=True)
    filepath = f"resultats/milp_TT/20j_5m/instance_{instance_id+1}.csv"

    with open(filepath, mode='w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames = ['instance', 'TT', 'status', 'gap(%)', 'cpu(s)']
        )
        writer.writeheader()
        if result:
            writer.writerow({
                'instance': instance_id + 1,
                'TT':       result['TT'],
                'status':   result['status'],
                'gap(%)':   result.get('gap', 'N/A'),
                'cpu(s)':   result.get('cpu', 'N/A')
            })


    print(f"\n Tableau sauvegardé : {filepath}")