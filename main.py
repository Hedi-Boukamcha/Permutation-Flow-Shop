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
