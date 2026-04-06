import numpy as np
from src.plots import plot_gantt
from src.dd_generator import generate_due_dates_brah
from src.scheduler import compute_objectives
from src.data_loader import load_all, display_dataset, save_instances
from to_csv import save_to_csv




if __name__ == "__main__":
    datasets = load_all("data/taillard")

    # Sauvegarder en CSV
    #display_dataset(datasets)
    print(f"\n{'='*50}")
    print("Sauvegarde des fichiers CSV...")
    save_to_csv(datasets, output_dir="data")
    save_instances(datasets)
    print("Done !")

    inst = datasets["tai20j_5m"][0]

    pt        = inst['processing_times']
    due_dates = generate_due_dates_brah(inst, tau=2)
    n_jobs    = inst['n_jobs']

    # Poids aléatoires entre 1 et 11
    weights  = np.random.randint(1, 11, size=n_jobs)

    # Séquence identitaire pour tester
    sequence = list(range(n_jobs))

    # Calcul
    obj = compute_objectives(sequence, pt, due_dates, weights)

    print(f"Completion times : {obj['Cj']}")
    print(f"Due dates        : {due_dates[sequence]}")
    print(f"Tardiness Tj     : {obj['Tj']}")
    print(f"Indicator Uj     : {obj['Uj']}")
    print(f"Poids wj         : {weights[sequence]}")
    print(f"\nTT    : {obj['TT']}")
    print(f"TWT   : {obj['TWT']}")
    print(f"T_max : {obj['T_max']}")
    print(f"NT    : {obj['NT']}")

    plot_gantt(
    sequence         = sequence,
    processing_times = pt,
    due_dates        = due_dates,
    weights          = weights,
    title            = "PFSP — Instance 1 tai20j_5m",
    filename         = "instance1_tai20j_5m.png"
)
