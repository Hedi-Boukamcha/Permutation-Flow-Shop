from src.data_loader import load_all

if __name__ == "__main__":
    datasets = load_all("data/taillard")

    for name, instances in datasets.items():
        print(f"\n{'='*40}")
        print(f"Dataset : {name}")
        print(f"Nombre d'instances : {len(instances)}")

        for idx, inst in enumerate(instances):
            print(f"\n  Instance {idx+1}:")
            print(f"    Jobs     : {inst['n_jobs']}")
            print(f"    Machines : {inst['n_machines']}")
            print(f"    UB       : {inst['ub']}")
            print(f"    LB       : {inst['lb']}")
            print(f"    Shape    : {inst['processing_times'].shape}")
            print(f"    PT[0]    : {inst['processing_times'][0]}")  # 1ère ligne (machine 1)