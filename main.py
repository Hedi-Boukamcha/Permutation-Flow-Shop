import numpy as np
from src.initial_solution import nehEdd, nehedd, taillard_sequences
from src.plots import plot_gantt
from src.dd_generator import generate_due_dates_brah, generate_weights
from src.scheduler import compute_objectives
from src.data_loader import load_all, display_dataset, save_instances
from to_csv import save_to_csv
from src.IG_TS_approche import destruction, reconstruction





if __name__ == "__main__":
    datasets = load_all("data/taillard")

    # Sauvegarder en CSV
    #display_dataset(datasets)
    print(f"\n{'='*50}")
    print("Sauvegarde des fichiers CSV...")
    save_to_csv(datasets, output_dir="data")
    save_instances(datasets)
    print("Done !")

    # ── Séquences identitaires ─────────────────────────────
    taillard_sequences(datasets)

    # ── NEHedd ─────────────────────────────────────────────
    nehEdd(datasets) 
    print("Done !")   


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

