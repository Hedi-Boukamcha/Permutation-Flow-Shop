import os
import sys
import numpy as np
from src.data_loader import load_all, load_instance
from src.dd_generator import generate_due_dates_brah
from src.math_model import solve_milp_tt

if __name__ == "__main__":
    # Récupérer les arguments
    subdir        = sys.argv[1]  # '20j_5m' ou '50j_10m'
    instance_file = sys.argv[2]  # 'instance_1.csv', 'instance_2.csv', etc.

    print(f"{'='*50}")
    print(f"Dataset  : {subdir}")
    print(f"Instance : {instance_file}")
    print(f"{'='*50}")

    # Charger l'instance
    instance_path = os.path.join("data/instances", subdir, instance_file)
    instance = load_instance(instance_path)
    
    pt = instance['processing_times']
    due_dates = instance['due_date']
    
    # Dossier résultats
    results_dir_milp = 'results/milp_tt'
    os.makedirs(os.path.join(results_dir_milp, subdir), exist_ok=True)
    result_file = os.path.join(results_dir_milp, subdir, instance_file)
    
    # Résoudre avec MILP
    print(f"\nRésolution MILP...")
    result = solve_milp_tt(pt.T, due_dates, time_limit=4*24*3600, filepath=result_file)
    
    if result:
        print(f"\n{'='*50}")
        print(f"Résultat final :")
        print(f"  Status      : {result['status']}")
        print(f"  TT          : {result['TT']}")
        print(f"  Temps (s)   : {result['time']:.2f}")
        print(f"  Gap (%)     : {result['gap']:.2f}")
        print(f"  Séquence    : {[j+1 for j in result['sequence']]}")
        print(f"  Fichier CSV : {result_file}")
    else:
        print("  Pas de solution trouvée")
