import csv
import logging
import os
import sys
import time
import numpy as np
from src.m import solve
from src.my_heur import heuristic_due_date_pfsp
from src.IG_TS_approche_v2 import IG_1F
from src.NEHedd_TB1 import results_nehedd_it1, run_nehedd_it1
from src.position_model import solve_milp_cmax
from src.TM_IG import run_tmig, tmig_wrapper
from src.GA_PathR import ga_pr_wrapper
from src.riahi_IGA import iga_riahi_final, run_on_instances
from src.NEHedd_FV import nehedd_tbit1, run_nehedd_FV
from src.initial_solution import nehEdd, nehedd, run_nehedd, taillard_sequences
from src.plots import plot_gantt, run_plots
from src.dd_generator import generate_due_dates_brah, generate_weights, load_weights
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


def save_summary_result(summary_csv, dataset_name, instance_file, result):
    output_dir = os.path.dirname(summary_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_exists = os.path.isfile(summary_csv)

    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Dataset", "Instance", "Status", "TT", "Temps (s)",
                "Gap (%)", "BestBound", "Objective", "Sequence"
            ])

        writer.writerow([
            dataset_name,
            instance_file,
            result["status"],
            result["TT"],
            #f"{result['time']:.4f}" if result["time"] != "" else "",
            f"{result['time']:.4f}" if result["time"] is not None else "",
            #f"{result['gap']:.2f}" if result["gap"] != "" else "",
            f"{result['gap']:.2f}" if result["gap"] is not None else "",
            result["best_bound"],
            result["objective_value"],
            #" ".join(str(j + 1) for j in result["sequence"]) if result["sequence"] else ""
            " ".join(str(j + 1) for j in result["sequence"]) if result["sequence"] is not None else ""
        ])

        f.flush()
        os.fsync(f.fileno())

def save_summary_result_heuristic(summary_csv, subdir, instance_file,  objective,result):

    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    existing_rows = []
    if os.path.isfile(summary_csv):
        with open(summary_csv, mode='r', newline='') as f:
            reader = csv.reader(f)
            existing_rows = list(reader)

    header = ["subdir", "instance", "TT", "TWT", "T_max", "NT", "time"]

    # garder header + lignes différentes de l'instance actuelle
    filtered_rows = [header]
    if existing_rows:
        start_idx = 1 if existing_rows[0] == header else 0
        for row in existing_rows[start_idx:]:
            if len(row) >= 2 and not (row[0] == subdir and row[1] == instance_file):
                filtered_rows.append(row)

    # ajouter la ligne courante
    filtered_rows.append([
        subdir,
        instance_file,
        result.get("TT", ""),
        result.get("TWT", ""),
        result.get("T_max", ""),
        result.get("NT", ""),
        round(result.get("time", 0), 4)
    ])

    with open(summary_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_rows)

def save_summary_result_by_objective(summary_csv, subdir, instance_file, result):
    import csv
    import os

    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    header = ["subdir", "instance", "TT", "TWT", "T_max", "NT", "time"]
    rows = []

    # lire l'existant
    if os.path.isfile(summary_csv):
        with open(summary_csv, mode='r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

    # garder seulement les lignes qui ne correspondent pas à cette instance
    filtered_rows = [header]

    if rows:
        start_idx = 1 if rows[0] == header else 0
        for row in rows[start_idx:]:
            if len(row) >= 2 and not (row[0] == subdir and row[1] == instance_file):
                filtered_rows.append(row)

    # ajouter la nouvelle ligne
    filtered_rows.append([
        subdir,
        instance_file,
        result.get("TT", ""),
        result.get("TWT", ""),
        result.get("T_max", ""),
        result.get("NT", ""),
        round(result.get("time", 0), 4)
    ])

    # réécrire tout le fichier
    with open(summary_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_rows)

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
    objectives = ['TT', 'TWT', 'T_max', 'NT']
    ordered_subdirs = ['20j_5m', '50j_10m']
    results_dir_nehedd = 'resultats/nehedd'
    results_dir_nehedd_it1 = 'results/nehedd_it1'
    results_dir_milp = 'results/milp_tt'
    results_dir_m = 'results/m_tt'
    results_dir_ig = 'resultats/ig_ts_v2'
    results_dir_tmig = 'results/tm_ig'
    results_dir_heur = 'results/my_heuristic'

    print("\n=== DEBUT DU JOB GLOBAL ===", flush=True)

    # ─────────────────────────────────────────────────────────
    # Data: toutes les instances
    # ─────────────────────────────────────────────────────────
    for subdir in ordered_subdirs:

        subdir_path = os.path.join(instances_dir, subdir)

        if not os.path.isdir(subdir_path):
            print(f"[SKIP] Dossier introuvable : {subdir_path}", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[DATASET] {subdir}", flush=True)
        print(f"{'='*60}", flush=True)

        summary_csv = os.path.join(results_dir_milp, subdir, "summary_results.csv")

        instance_files = sorted([f for f in os.listdir(subdir_path) if f.endswith(".csv")])

        for idx, instance_file in enumerate(instance_files, start=1):

            print(f"\n[INSTANCE {idx}/{len(instance_files)}] {instance_file}", flush=True)

            instance_path = os.path.join(subdir_path, instance_file) 
            instance = load_instance(instance_path)
###########################################################

            pt = instance['processing_times']
            due_date = instance['due_date']

            # ─────────────────────────────────────────────────────────
            # EXECUTION 1 : Modele Math
            # ─────────────────────────────────────────────────────────
            print(f"[RUN] MILP (OR-Tools) pour {subdir}_{instance_file}", flush=True)

            result_file = os.path.join(results_dir_milp, subdir, instance_file)

            result = solve_milp_tt(
                pt,
                due_date,
                time_limit=600,
                filepath=result_file,
                instance_name=f"{subdir}/{instance_file}",
                use_heuristic=None
            )

            # sauvegarde immédiate dans le résumé global
            save_summary_result(summary_csv, subdir, instance_file, result)
            print(f"[SAVE] Résumé global mis à jour : {summary_csv}", flush=True)

            if result and result["sequence"]:
                print(f"  Séquence : {[j+1 for j in result['sequence']]}", flush=True)
                print(f"  Tardiness : {result['TT']}", flush=True)
                if result["gap"] != "":
                    print(f"  Gap : {result['gap']:.2f}%", flush=True)
            else:
                print("  Pas de solution trouvée", flush=True)

            print("\n=== FIN DU JOB GLOBAL ===", flush=True)

            

            summary_csv_heur = os.path.join(results_dir_heur, "summary_heuristic.csv")
            summary_csv_nehedd = os.path.join(results_dir_nehedd, "summary_nehedd.csv")
            summary_csv_nehedd_it1 = os.path.join(results_dir_nehedd_it1, "summary_nehedd_it1.csv")
            

            print(f"[RUN] Heuristic pour {subdir}_{instance_file}", flush=True)

            instance_name = instance_file.replace(".csv", "")
            heur_file = os.path.join(results_dir_heur, subdir, f"{instance_name}_TT.csv")
            weights_path = os.path.join("data/weights", f"{subdir}_weights.csv")
            weights = load_weights(weights_path)
            due_date = instance['due_date']
            heur_results = {}
            nehedd_results = {}
            nehedd_it1_results = {}
            tmig_results = {}
            

            for obj in objectives:
                            # ─────────────────────────────────────────────────────────
                            # EXECUTION ++++++++++++++++++ : Mon Heuristique
                            # ─────────────────────────────────────────────────────────
                """print(f"[RUN] Heuristic ({obj}) pour {subdir}_{instance_file}", flush=True)

                heur_file = os.path.join(
                    results_dir_heur,
                    subdir,
                    f"{instance_name}_{obj}.csv"
                )

                heur_result = heuristic_due_date_pfsp(
                    instance=instance,
                    weights=weights,
                    objective=obj,
                    k=4,
                    max_iter=10,
                    filepath=heur_file,
                    verbose=False
                )

                heur_results[obj] = heur_result

                if obj == "TT" and heur_result and heur_result["sequence"]:
                    gantt_file = os.path.join(
                        results_dir_heur,
                        subdir,
                        "gantts",
                        f"{instance_name}_TT_gantt.png"
                    )

                    plot_gantt(
                        sequence=heur_result["sequence"],
                        processing_times=instance["processing_times"],
                        due_dates=due_date,
                        weights=weights,
                        objective="TT",
                        title=f"Gantt - TT - {instance_name}",
                        filename=gantt_file
                    )

                print(f"[DEBUG] summary_csv_nehedd = {summary_csv_heur}", flush=True)

                summary_csv_heur = os.path.join(
                    results_dir_heur,
                    subdir,
                    f"summary_heuristic_{obj}.csv"
                )

                save_summary_result_by_objective(
                    summary_csv_heur,
                    subdir,
                    instance_file,
                    heur_result
                )

                print(
                    f"  {obj} -> TT={heur_result['TT']}, "
                    f"TWT={heur_result['TWT']}, "
                    f"T_max={heur_result['T_max']}, "
                    f"NT={heur_result['NT']}, "
                    f"Time={heur_result['time']:.2f}s",
                    flush=True
                )

                print(f"  {obj} -> TT={heur_result['TT']}, TWT={heur_result['TWT']}, "
                f"T_max={heur_result['T_max']}, NT={heur_result['NT']}, "
                f"Time={heur_result['time']:.2f}s", flush=True)

                # si tu veux un résumé séparé pour l’heuristique
                #save_summary_result_heuristic(summary_csv_heur, subdir, instance_file, heur_result)
                
                #print(f"[SAVE] Résumé heuristique mis à jour : {summary_csv_heur}", flush=True)

                if heur_result and heur_result["sequence"]:
                    print(f"  Séquence heuristique : {[j+1 for j in heur_result['sequence']]}", flush=True)
                    print(f"  TT heuristique : {heur_result['TT']}", flush=True)
                    print(f"  Temps heuristique : {heur_result['time']:.2f}s", flush=True)
                #else:"""


                

                            # ─────────────────────────────────────────────────────────
                            # EXECUTION ++++++++++++++++++ : NEH EDD
                            # ─────────────────────────────────────────────────────────
            
                """print(f"[RUN] NEHedd ({obj}) pour {subdir}_{instance_file}", flush=True)
                
                nehedd_file = os.path.join(
                    results_dir_nehedd,
                    subdir,
                    f"{instance_name}_{obj}.csv"
                )

                nehedd_result = run_nehedd(
                    instance=instance,
                    due_dates=due_date,
                    weights=weights,
                    objective=obj,
                    filepath=nehedd_file
                )

                if obj == "TT" and nehedd_result and nehedd_result["sequence"]:
                    gantt_file = os.path.join(
                        results_dir_nehedd_it1,
                        subdir,
                        "gantts",
                        f"{instance_name}_TT_gantt.png"
                    )

                    plot_gantt(
                        sequence=nehedd_result["sequence"],
                        processing_times=instance["processing_times"],
                        due_dates=due_date,
                        weights=weights,
                        objective="TT",
                        title=f"Gantt - TT - {instance_name}",
                        filename=gantt_file
                    )

                summary_csv_nehedd = os.path.join(
                    results_dir_nehedd,
                    subdir,
                    f"summary_{obj}.csv"
                )

                print(f"[DEBUG] summary_csv_nehedd = {summary_csv_nehedd}", flush=True)

                save_summary_result_by_objective(
                    summary_csv_nehedd,
                    subdir,
                    instance_file,
                    nehedd_result
                )

                print(
                    f"  {obj} -> TT={nehedd_result['TT']}, "
                    f"TWT={nehedd_result['TWT']}, "
                    f"T_max={nehedd_result['T_max']}, "
                    f"NT={nehedd_result['NT']}, "
                    f"Time={nehedd_result['time']:.2f}s",
                    flush=True
                )"""

                            # ─────────────────────────────────────────────────────────
                            # EXECUTION ++++++++++++++++++ : NEH EDD IT1
                            # ─────────────────────────────────────────────────────────


                """print(f"[RUN] NEHedd_IT1 ({obj}) pour {subdir}_{instance_file}", flush=True)

                nehedd_it1_file = os.path.join(
                    results_dir_nehedd_it1,
                    subdir,
                    f"{instance_name}_NEHedd_IT1_{obj}.csv"
                )

                result_it1 = run_nehedd_it1(
                    instance=instance,
                    weights=weights,
                    objective=obj,
                    filepath=nehedd_it1_file
                )

                nehedd_it1_results[obj] = result_it1

                if obj == "TT" and result_it1 and result_it1["sequence"]:
                    gantt_file = os.path.join(
                        results_dir_nehedd_it1,
                        subdir,
                        "gantts",
                        f"{instance_name}_TT_gantt.png"
                    )

                    plot_gantt(
                        sequence=result_it1["sequence"],
                        processing_times=instance["processing_times"],
                        due_dates=due_date,
                        weights=weights,
                        objective="TT",
                        title=f"Gantt - TT - {instance_name}",
                        filename=gantt_file
                    )

                summary_csv_it1 = os.path.join(
                    results_dir_nehedd_it1,
                    subdir,
                    f"summary_NEHedd_IT1_{obj}.csv"
                )

                save_summary_result_by_objective(
                    summary_csv_it1,
                    subdir,
                    instance_file,
                    result_it1
                )

                print(
                    f"  {obj} -> TT={result_it1['TT']}, "
                    f"TWT={result_it1['TWT']}, "
                    f"T_max={result_it1['T_max']}, "
                    f"NT={result_it1['NT']}, "
                    f"Time={result_it1['time']:.2f}s, "
                    f"Ties={result_it1['total_ties']}",
                    flush=True
                )"""                
                print("\n=== FIN INSTANCE ===", flush=True)             


                            # ─────────────────────────────────────────────────────────
                            # EXECUTION ++++++++++++++++++ : TM_IG tabu mem + IG
                            # ─────────────────────────────────────────────────────────

            """print(f"[RUN] TM-IG ({obj}) pour {subdir}_{instance_file}", flush=True)

                tmig_file = os.path.join(
                    results_dir_tmig,
                    subdir,
                    f"{instance_name}_{obj}.csv"
                )

                tmig_result = run_tmig(
                    instance=instance,
                    weights=weights,
                    objective=obj,
                    filepath=tmig_file
                )

                tmig_results[obj] = tmig_result

                summary_csv_tmig = os.path.join(
                    results_dir_tmig,
                    subdir,
                    f"summary_{obj}.csv"
                )

                save_summary_result_by_objective(
                    summary_csv_tmig,
                    subdir,
                    instance_file,
                    tmig_result
                )

                print(
                    f"  {obj} -> TT={tmig_result['TT']}, "
                    f"TWT={tmig_result['TWT']}, "
                    f"T_max={tmig_result['T_max']}, "
                    f"NT={tmig_result['NT']}, "
                    f"Time={tmig_result['time']:.2f}s",
                    flush=True
                )"""

#########################################

            # ─────────────────────────────────────────────────────────
            # EXECUTION 2 : IG_TS_V2: 
            # A tabu memory based iterated greedy algorithm 
            # for the distributed heterogeneous permutation flowshop scheduling problem 
            # with the total tardiness criterion
            # ─────────────────────────────────────────────────────────

"""print("Shape pt:", pt.shape)

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
    print(f"  Temps = {result_ig['time']:.2f}s")"""


 

