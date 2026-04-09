import csv
import os
import sys
import numpy as np
from src.position_model import solve_milp_cmax
from src.TM_IG import tmig_wrapper
from src.GA_PathR import ga_pr_wrapper
from src.riahi_IGA import iga_riahi_final, run_on_instances
from src.NEHedd_FV import nehedd_tbit1, run_nehedd_FV, save_results
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
  # ─────────────────────────────────────────────────────────
    # EXECUTION 1 : NEH_EDD (Fernandez-Viagas & Framinan)
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RUNNING: NEHedd — Tie-Breaking IT1")
    print("=" * 55)
    
    out_neh   = "resultats/nehedd_FV"
    out_iga   = "resultats/iga_riahi"
    out_ga_pr = "resultats/ga_pr"
    out_tmig = "resultats/tmig"
    os.makedirs(out_neh,   exist_ok=True)
    os.makedirs(out_iga,   exist_ok=True)
    os.makedirs(out_ga_pr, exist_ok=True)
    os.makedirs(out_tmig, exist_ok=True)


    # 3. Boucle sur les datasets
    """for name, instances in datasets.items():
        if name in folder_map:
            label = folder_map[name]

            # --- EXECUTION NEH ---
            print(f"\nLancement NEHedd sur {name}...")
            res_neh = run_on_instances(name, instances, nehedd_tbit1)
            save_results(res_neh, os.path.join(out_neh, f"{label}_results.csv"))

            # --- EXECUTION IGA ---
            print(f"\nLancement IGA Riahi sur {name}...")
            res_iga = run_on_instances(name, instances, iga_riahi_final)
            save_results(res_iga, os.path.join(out_iga, f"{label}_results.csv"))

            # --- EXECUTION GA-PR ---
            print(f"\nLancement GA-PR sur {name}...")
            res_ga_pr = run_on_instances(name, instances, ga_pr_wrapper)
            save_results(res_ga_pr, os.path.join(out_ga_pr, f"{label}_results.csv"))

            print(f"\nLancement TM-IG sur {name}...")
            res_tmig = run_on_instances(name, instances, tmig_wrapper)
            save_results(res_tmig, os.path.join(out_tmig, f"{label}_results.csv"))

    print("\n[OK] Tous les calculs sont terminés.")""" 

    instances = datasets["tai20j_5m"]
    rows      = []

    for idx, inst in enumerate(instances):
        pt        = inst['processing_times']
        due_dates = generate_due_dates_brah(inst, tau=2)

        print(f"\n{'='*50}")
        print(f"Instance {idx+1} — tai20j_5m")

        result = solve_milp_cmax(
            processing_times = pt,
            due_dates        = due_dates,
            #filepath         = None,  # ← pas de fichier par instance
            time_limit       = 600
        )

        if result:
            rows.append({
                'instance': idx + 1,
                'TT':       result['TT'],
                'status':   result['status'],
                'gap(%)':   result.get('gap', 'N/A'),
                'cpu(s)':   result.get('cpu', 'N/A')
            })
        """else:
            rows.append({
                'instance': idx + 1,
                'TT':       'N/A',
                'status':   'NO SOLUTION',
                'gap(%)':   'N/A',
                'cpu(s)':   'N/A'
            })"""

    # ── Un seul fichier pour toutes les instances ─────────
    os.makedirs("resultats/milp_new", exist_ok=True)
    filepath = "resultats/milp_new/20j_5m_testcmax.csv"

    with open(filepath, mode='w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames = ['instance', 'TT', 'status', 'gap(%)', 'cpu(s)']
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n Tableau sauvegardé : {filepath}")



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
 

