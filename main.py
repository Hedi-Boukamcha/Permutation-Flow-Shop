import os
import sys
import numpy as np
from src.NEHedd_FV import run_dataset, save_results
from src.initial_solution import nehEdd, nehedd, taillard_sequences
from src.plots import plot_gantt, run_plots
from src.dd_generator import generate_due_dates_brah, generate_weights
from src.scheduler import compute_objectives
from src.data_loader import load_all, display_dataset, save_instances
from to_csv import save_to_csv
from src.IG_TS_approche import runIG, destruction, reconstruction, runIG, runIG_TS
from src.math_model import solve_milp_tt





if __name__ == "__main__":
    datasets = load_all("data/taillard")
    
    folder_map = {
        "tai20j_5m":  "20j_5m",
        "tai50j_10m": "50j_10m",
    }

# ─────────────────────────────────────────────────────────
# NEH_EDD - NEH-based heuristics for the permutation flowshop scheduling problem to minimise total tardiness
# Victor Fernandez-Viagas1†, Jose M. Framinan1
# ─────────────────────────────────────────────────────────

    print("=" * 55)
    print("  NEHedd — Tie-Breaking IT1 (Fernandez-Viagas 2015)")
    print("  Objectif : minimisation TT")
    print("=" * 55)
    output_dir="resultats/nehedd_FV"
    # 2. Définir les alias pour les noms de fichiers


    # 3. Boucler sur les dossiers trouvés
    for name, instances in datasets.items():
        # Lancer le calcul (défini dans heuristics.py)
        results = run_dataset(name, instances)
        # Déterminer le nom du fichier CSV
        label    = folder_map.get(name, name)
        filepath = os.path.join(output_dir, f"{label}_results.csv")
        # Sauvegarder
        save_results(results, filepath)
    print("\n[OK] Tous les calculs sont terminés.")

    # Sauvegarder en CSV
    #display_dataset(datasets)
    """print(f"\n{'='*50}")
    print("Sauvegarde des fichiers CSV...")
    save_to_csv(datasets, output_dir="data")
    save_instances(datasets)
    print("Done !")

    # ── Séquences identitaires ─────────────────────────────
    taillard_sequences(datasets)

    # ── NEHedd ─────────────────────────────────────────────
    nehEdd(datasets) 

    runIG(datasets)

# Ou toutes les instances
    runIG_TS(datasets)

    run_plots(datasets)"""

"""        # Récupérer les arguments
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
    result = solve_milp_tt(
        processing_times = pt,
        due_dates        = due_dates,
        time_limit       = 3600,   # 1 heure
        filepath         = filepath
    )

    if result:
        print(f"\nRésultat final :")
        print(f"  Status   : {result['status']}")
        print(f"  TT       : {result['TT']}")
        print(f"  Séquence : {[j+1 for j in result['sequence']]}")
    print("Done !")  """ 

    
""" 

    # ── Charger une instance ────────────────────────────────
    inst      = datasets["tai20j_5m"][0]
    pt        = inst['processing_times']
    due_dates = generate_due_dates_brah(inst, tau=2)
    weights   = generate_weights(inst)
    n_jobs    = inst['n_jobs']

    sequence = nehedd(pt, due_dates, weights, objective='TT')
    print(f"Séquence initiale : {[j+1 for j in sequence]}")

    # ── Test destruction ────────────────────────────────────
    partial_seq, removed = destruction(
        sequence         = sequence,
        processing_times = pt,
        due_dates        = due_dates,
        weights          = weights,
        k                = 4,
        objective        = 'TT'
    )
    print(f"Séquence partielle : {[j+1 for j in partial_seq]}")
    print(f"Jobs retirés       : {[j+1 for j in removed]}")

    # ── Test reconstruction ─────────────────────────────────
    new_seq = reconstruction(
        partial_sequence = partial_seq,
        removed_jobs     = removed,
        processing_times = pt,
        due_dates        = due_dates,
        weights          = weights,
        objective        = 'TT'
    )
    print(f"Séquence reconstruite : {[j+1 for j in new_seq]}")

    # ── Comparer avant/après ────────────────────────────────
    obj_avant = compute_objectives(sequence, pt, due_dates, weights)
    obj_apres = compute_objectives(new_seq,  pt, due_dates, weights)

    print(f"\n{'='*40}")
    print(f"{'':15} {'Avant':>10} {'Après':>10}")
    print(f"{'='*40}")
    print(f"{'TT':15} {obj_avant['TT']:>10} {obj_apres['TT']:>10}")
    print(f"{'TWT':15} {obj_avant['TWT']:>10} {obj_apres['TWT']:>10}")
    print(f"{'T_max':15} {obj_avant['T_max']:>10} {obj_apres['T_max']:>10}")
    print(f"{'NT':15} {obj_avant['NT']:>10} {obj_apres['NT']:>10}")
    print(f"{'='*40}")

    for objective in ['TT', 'TWT', 'T_max', 'NT']:
        print(f"\n{'='*40}")
        print(f"Objectif : {objective}")

        best_seq, best_val, history = IG(
            processing_times = pt,
            due_dates        = due_dates,
            weights          = weights,
            objective        = objective,
            k                = 4,
            max_iter         = 100,
            filepath         = f"resultats/ig/20j_5m/instance_1_{objective}.csv"
        )

        obj = compute_objectives(best_seq, pt, due_dates, weights)
        print(f"  Séquence : {[j+1 for j in best_seq]}")
        print(f"  TT       : {obj['TT']}")
        print(f"  TWT      : {obj['TWT']}")
        print(f"  T_max    : {obj['T_max']}")
        print(f"  NT       : {obj['NT']}")"""
 

