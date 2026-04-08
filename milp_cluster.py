import sys
import numpy as np
from src.data_loader import load_all
from src.dd_generator import generate_due_dates_brah
from src.math_model import solve_milp_tt

if __name__ == "__main__":

    # Récupérer les arguments
    instance_id  = int(sys.argv[1])   # 0 à 9
    dataset_name = sys.argv[2]         # tai20j_5m ou tai50j_10m

    print(f"{'='*50}")
    print(f"Dataset  : {dataset_name}")
    print(f"Instance : {instance_id + 1}")
    print(f"{'='*50}")

    # Charger les données
    datasets  = load_all("data/taillard")
    inst      = datasets[dataset_name][instance_id]
    pt        = inst['processing_times']
    due_dates = generate_due_dates_brah(inst, tau=2)

    # Dossier résultats
    folder_map = {
        "tai20j_5m":  "20j_5m",
        "tai50j_10m": "50j_10m"
    }
    folder   = folder_map.get(dataset_name, dataset_name)
    filepath = f"resultats/milp/{folder}/instance_{instance_id+1}.csv"

    # Résoudre avec MILP
    print(f"\nRésolution MILP...")
    result = solve_milp_tt(
        processing_times = pt,
        due_dates        = due_dates,
        time_limit       = 3600,
        filepath         = filepath
    )

    if result:
        print(f"\n{'='*50}")
        print(f"Résultat final :")
        print(f"  Status   : {result['status']}")
        print(f"  TT       : {result['TT']}")
        print(f"  Séquence : {[j+1 for j in result['sequence']]}")
        print(f"  Fichier  : {filepath}")