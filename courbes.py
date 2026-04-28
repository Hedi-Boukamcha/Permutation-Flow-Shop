import os
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

OBJECTIVES = ["TT", "NT", "TWT", "T_max"]  # ajoute "TWT", "T_max" si besoin

SIZES = ["20j_5m", "50j_10m"]

METHODS = {
    "IG_MS": "results/my_heuristic/{size}/summary_heuristic_{obj}.csv",
    "IG_tb": "results/tm_ig/{size}/summary_{obj}.csv",
    "NEHEDD_(TBIT1)": "results/nehedd_it1/{size}/summary_NEHedd_IT1_{obj}.csv",
}

OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_col(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def load_results(method, size, obj):
    path = METHODS[method].format(size=size, obj=obj)

    if not os.path.exists(path):
        print(f"[WARNING] Fichier introuvable : {path}")
        return None

    df = pd.read_csv(path)

    instance_col = find_col(df, [
        "Instance", "instance", "Instance file",
        "instance_file", "File", "file"
    ])

    obj_col = find_col(df, [
        obj, obj.lower(), "Objective", "objective", "Value", "value"
    ])

    if instance_col is None:
        raise ValueError(f"Colonne instance introuvable dans {path}")

    if obj_col is None:
        raise ValueError(f"Colonne objectif {obj} introuvable dans {path}")

    tmp = df[[instance_col, obj_col]].copy()
    tmp = tmp.rename(columns={
        instance_col: "Instance",
        obj_col: "Objective"
    })

    tmp["Instance"] = tmp["Instance"].astype(str).str.extract(r"(\d+)").astype(int)
    tmp = tmp.sort_values("Instance")

    return tmp


def plot_objective(obj):
    fig, axes = plt.subplots(1, len(SIZES), figsize=(12, 4), sharey=False)

    if len(SIZES) == 1:
        axes = [axes]

    for ax, size in zip(axes, SIZES):
        for method in METHODS.keys():
            df = load_results(method, size, obj)

            if df is None:
                continue

            ax.plot(
                df["Instance"],
                df["Objective"],
                marker="o",
                linewidth=1.5,
                label=method
            )

        ax.set_title(f"{obj} - {size}")
        ax.set_xlabel("Instance")
        ax.set_ylabel(obj)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"evolution_{obj}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] Figure créée : {output_path}")


def main():
    for obj in OBJECTIVES:
        plot_objective(obj)


if __name__ == "__main__":
    main()