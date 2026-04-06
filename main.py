import numpy as np
from src.initial_solution import nehEdd, taillard_sequences
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

    # ── Séquences identitaires ─────────────────────────────
    taillard_sequences(datasets)

    # ── NEHedd ─────────────────────────────────────────────
    nehEdd(datasets) 
    print("Done !")   

