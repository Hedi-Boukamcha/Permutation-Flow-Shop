import csv
import os
import sys
import numpy as np
from src.IG_TS_approche_v2 import IG_1F
from src.NEHedd_TB1 import results_nehedd_it1
from src.position_model import solve_milp_cmax
from src.TM_IG import tmig_wrapper
from src.GA_PathR import ga_pr_wrapper
from src.riahi_IGA import iga_riahi_final, run_on_instances
from src.NEHedd_FV import nehedd_tbit1, run_nehedd_FV
from src.initial_solution import nehEdd, nehedd, taillard_sequences
from src.plots import plot_gantt, run_plots
from src.dd_generator import generate_due_dates_brah, generate_weights
from src.scheduler import compute_objectives
from src.data_loader import load_all, display_dataset, load_instance, save_instances
from to_csv import save_to_csv
from src.IG_TS_approche import runIG, destruction, reconstruction, runIG, runIG_TS
from src.math_model import solve_milp_tt

# Commande CC
# Consulter les erruers: cat id_job_err.txt
# modifier un fichier : nano nom_fichier
# Lancer les jobs: sbatch job.sh
# Consulter les jobs: squeue -u nom_user



if __name__ == "__main__":
    #datasets = load_all("data/taillard")
    
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
    
    """    out_neh   = "resultats/nehedd_FV"
    out_iga   = "resultats/iga_riahi"
    out_ga_pr = "resultats/ga_pr"
    out_tmig = "resultats/tmig"
    os.makedirs(out_neh,   exist_ok=True)
    os.makedirs(out_iga,   exist_ok=True)
    os.makedirs(out_ga_pr, exist_ok=True)
    os.makedirs(out_tmig, exist_ok=True)"""

    """    instances_dirs = [
        "data/instances/20j_5m",
        "data/instances/50j_10m"
    ]    

    print("Exécution de l'heuristique NEHedd_IT1")
    instances_dir = "data/instances"
    results_dir = './resultats/nehedd_it1'
    results_nehedd_it1(instances_dir, output_dir=results_dir)
    print(f"Résultats enregistrés dans {os.path.abspath(results_dir)}")"""

    """base_path   = "data/instances"
    results_dir = "./resultats"
    os.makedirs(results_dir, exist_ok=True)

    # ─────────────────────────────
    # 🔹 MÉTHODE 1 : run_nehedd_FV
    # ─────────────────────────────
    print("\n--- Méthode 1 : run_nehedd_FV ---")

    datasets = ["20j_5m", "50j_10m"]

    for dataset in datasets:

        path = os.path.join(base_path, dataset)

        # charger les instances
        instances = load_all(path)

        # exécuter
        results = run_nehedd_FV(dataset, instances)

        # sauvegarder
        output_file = os.path.join(results_dir, f"{dataset}_FV.csv")
        save_results(results, output_file)


    # ─────────────────────────────
    # 🔹 MÉTHODE 2 : results_nehedd_it1
    # ─────────────────────────────
    print("\n--- Méthode 2 : results_nehedd_it1 ---")

    results_nehedd_it1(base_path, output_dir=os.path.join(results_dir, "nehedd_it1"))


    print(f"\nRésultats enregistrés dans {os.path.abspath(results_dir)}")"""


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

    """instances = datasets["tai20j_5m"]
    rows      = []

    for idx, inst in enumerate(instances):
        #pt        = inst['processing_times']
        due_dates = generate_due_dates_brah(inst, tau=2)

        print(f"\n{'='*50}")
        print(f"Instance {idx+1} — tai20j_5m")
       
        inst      = datasets["tai20j_5m"][0]
        pt        = inst['processing_times']

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


    # ── Un seul fichier pour toutes les instances ─────────
    os.makedirs("resultats/milp_TT", exist_ok=True)
    filepath = "resultats/milp_TT/20j_5m_testcmax.csv"

    with open(filepath, mode='w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames = ['instance', 'TT', 'status', 'gap(%)', 'cpu(s)']
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n Tableau sauvegardé : {filepath}")"""

    




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

    instances_dir = "data/instances"
    ordered_subdirs = ['20j_5m', '50j_10m']
    results_dir_nehedd = 'results/nehedd_it1'
    results_dir_milp = 'results/milp_tt'
    results_dir_ig = 'resultats/ig_ts_v2'

    # ─────────────────────────────────────────────────────────
    # Data: toutes les instances
    # ─────────────────────────────────────────────────────────
    for subdir in ordered_subdirs:

        subdir_path = os.path.join(instances_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        print(f"\n{'='*50}")
        print(f"Dataset : {subdir}")

        # Parcours des fichiers instances
        for instance_file in sorted(os.listdir(subdir_path)):

            if not instance_file.endswith(".csv"):
                continue

            print(f"\nInstance : {instance_file}")

            instance_path = os.path.join(subdir_path, instance_file)
            instance = load_instance(instance_path)

            pt = instance['processing_times']
            due_dates = instance['due_date']

            # ─────────────────────────────────────────────────────────
            # EXECUTION 1 : Modele Math
            # ─────────────────────────────────────────────────────────

            """print(f"\nExécution de MILP (OR-Tools) pour l'instance {subdir}_{instance_file}")
            #time_limit = 60  # Temps limite en secondes
            result_file = os.path.join(results_dir_milp, subdir, instance_file)
            result = solve_milp_tt(pt.T, due_dates, time_limit = 3600, filepath=result_file)
            
            if result:
                print(f"  Séquence : {[j+1 for j in result['sequence']]}")
                print(f"  Tardiness : {result['TT']}")
            else:
                print("  Pas de solution trouvée")"""

            # ─────────────────────────────────────────────────────────
            # EXECUTION 2 : IG_TS_V2: 
            # A tabu memory based iterated greedy algorithm 
            # for the distributed heterogeneous permutation flowshop scheduling problem 
            # with the total tardiness criterion
            # ─────────────────────────────────────────────────────────

            print("Shape pt:", pt.shape)

            objectives = ['TT', 'TWT', 'T_max', 'NT']
            instance_name = instance_file.replace(".csv", "")

            for obj in objectives:

                print(f"\nExécution de IG ({obj}) pour l'instance {subdir}_{instance_file}")

                result_file_ig = os.path.join(
                    results_dir_ig,
                    subdir,
                    f"{instance_name}_{obj}.csv"
                )

                result_ig = IG_1F(
                    instance,
                    due_dates,
                    objective=obj,
                    k=4,
                    max_iter=10,
                    filepath=result_file_ig
                )

                print(f"  Séquence : {[j+1 for j in result_ig['sequence']]}")
                print(f"  TT   = {result_ig['TT']}")
                print(f"  TWT  = {result_ig['TWT']}")
                print(f"  T_max= {result_ig['T_max']}")
                print(f"  NT   = {result_ig['NT']}")
                print(f"  Temps = {result_ig['time']:.2f}s")

            # ─────────────────────────────────────────────────────────
            # EXECUTION 3 : NEHedd_IT1 
            # ─────────────────────────────────────────────────────────
            
            """print(f"\nExécution de NEHedd_IT1 pour l'instance {subdir}_{instance_file}")
            sequence, total_ties, elapsed = NEHedd_IT1(instance, due_dates)
            print(f"  Séquence : {[j+1 for j in sequence]}")
            print(f"  Total ties : {total_ties}")
            print(f"  Temps d'exécution : {elapsed:.2f}s")"""

    







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
 

