import random
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import os

from src.IG_TS_approche import ig, ig_ts
from src.dd_generator import generate_due_dates_brah, generate_weights
from src.initial_solution import nehedd
from src.scheduler import compute_completion_times, compute_objectives



def generate_colors(n_jobs, seed=42):
    """
    Génère n couleurs distinctes avec degré de visibilité fixe.
    Teinte aléatoire reproductible, saturation et luminosité fixes.
    """
    random.seed(seed)
    colors = []
    for _ in range(n_jobs):
        h = random.random()       # teinte aléatoire [0, 1]
        s = 0.70                  # saturation fixe  → couleurs vives
        l = 0.45                  # luminosité fixe  → bon contraste texte blanc

        # Conversion HSL → RGB
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        colors.append((r, g, b))
    return colors


def plot_gantt(
    sequence,
    processing_times,
    due_dates=None,
    weights=None,
    objective="TT",
    title=None,
    filename="gantt.png"
):
    """
    Gantt générique pour toutes les heuristiques.
    Affiche pour chaque opération :
    - le job
    - la date de début
    - la date de fin Cij

    sequence : liste des jobs indexés à partir de 0
    processing_times : matrice (n_machines, n_jobs)
    due_dates : dates dues des jobs
    objective : TT, TWT, T_max, NT...
    """

    n_machines = processing_times.shape[0]
    n_jobs = len(sequence)

    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    C = compute_completion_times(sequence, processing_times)

    starts = np.zeros((n_machines, n_jobs), dtype=int)

    for pos in range(n_jobs):
        for i in range(n_machines):
            if pos == 0 and i == 0:
                starts[i, pos] = 0
            elif i == 0:
                starts[i, pos] = C[i, pos - 1]
            elif pos == 0:
                starts[i, pos] = C[i - 1, pos]
            else:
                starts[i, pos] = max(C[i, pos - 1], C[i - 1, pos])

    colors = generate_colors(n_jobs, seed=42)

    if title is None:
        title = f"Gantt Chart - Objective {objective}"

    fig, ax = plt.subplots(figsize=(18, n_machines * 1.6 + 3))

    for pos, job in enumerate(sequence):
        for i in range(n_machines):
            start = starts[i, pos]
            end = C[i, pos]
            duration = end - start

            ax.barh(
                y=i,
                width=duration,
                left=start,
                height=0.6,
                color=colors[pos],
                edgecolor="white",
                linewidth=0.8
            )

            ax.text(
                x=start + duration / 2,
                y=i,
                s=f"J{job + 1}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold"
            )

    # Cj sur la dernière machine
    last_machine = n_machines - 1
    completion_times = C[last_machine]

    for pos, job in enumerate(sequence):
        cj = completion_times[pos]

        ax.axvline(
            x=cj,
            color=colors[pos],
            linestyle=":",
            linewidth=1,
            alpha=0.8
        )

        label = f"C{job + 1}={cj}"

        ax.text(
            x=cj,
            y=n_machines + 0.5,
            s=label,
            fontsize=6,
            color=colors[pos],
            ha="center",
            va="bottom",
            rotation=90
        )
    # ─── Start times sur chaque machine ─────────────────────
    for pos, job in enumerate(sequence):
        for i in range(n_machines):

            start = starts[i][pos]

            ax.axvline(
                x=start,
                color=colors[pos],
                linestyle=':',
                linewidth=0.5,
                alpha=0.3
            )

            # label uniquement pour éviter surcharge (option)
            if i == 0:  # seulement M1 (sinon trop chargé)
                ax.text(
                    x=start,
                    y=n_machines + 0.1,
                    s=f"S{job+1}={start}",
                    fontsize=6,
                    color=colors[pos],
                    ha='center',
                    va='bottom',
                    rotation=90
                )

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"Machine {i + 1}" for i in range(n_machines)])
    ax.invert_yaxis()

    ax.set_xlabel("Temps")
    ax.set_ylabel("Machines")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.text(
    x = start + duration / 2,
    y = i,
    s = label,
    ha = 'center',
    va = 'center',
    fontsize = 7,
    color = 'white',
    fontweight = 'bold'
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Gantt sauvegardé : {filename}")

    
def compute_all_results(datasets):
    """
    Calcule les résultats pour toutes les méthodes et instances.
    """
    results = {}

    for name, instances in datasets.items():
        results[name] = {
            'taillard': {'TT': [], 'TWT': [], 'T_max': [], 'NT': []},
            'nehedd':   {'TT': [], 'TWT': [], 'T_max': [], 'NT': []},
            'ig':       {'TT': [], 'TWT': [], 'T_max': [], 'NT': []},
            'ig_ts':    {'TT': [], 'TWT': [], 'T_max': [], 'NT': []}  # ← nouveau
        }

        for idx, inst in enumerate(instances):
            pt        = inst['processing_times']
            due_dates = generate_due_dates_brah(inst, tau=2)
            weights   = generate_weights(inst)
            n_jobs    = inst['n_jobs']

            print(f"  {name} — Instance {idx+1}...")

            # ── Taillard (séquence identitaire) ──────────────
            seq_tai = list(range(n_jobs))
            obj_tai = compute_objectives(seq_tai, pt, due_dates, weights)
            for obj in ['TT', 'TWT', 'T_max', 'NT']:
                results[name]['taillard'][obj].append(obj_tai[obj])

            # ── NEHedd ───────────────────────────────────────
            for obj in ['TT', 'TWT', 'T_max', 'NT']:
                seq_neh = nehedd(pt, due_dates, weights, objective=obj)
                obj_neh = compute_objectives(seq_neh, pt, due_dates, weights)
                results[name]['nehedd'][obj].append(obj_neh[obj])

            # ── IG ───────────────────────────────────────────
            for obj in ['TT', 'TWT', 'T_max', 'NT']:
                seq_ig, _, _ = ig(
                    processing_times = pt,
                    due_dates        = due_dates,
                    weights          = weights,
                    objective        = obj,
                    k                = 4,
                    max_iter         = 100
                )
                obj_ig = compute_objectives(seq_ig, pt, due_dates, weights)
                results[name]['ig'][obj].append(obj_ig[obj])

            # ── IG-TS ─────────────────────────────────────── ← nouveau
            for obj in ['TT', 'TWT', 'T_max', 'NT']:
                seq_igts, _, _ = ig_ts(
                    processing_times = pt,
                    due_dates        = due_dates,
                    weights          = weights,
                    objective        = obj,
                    k                = 4,
                    max_iter         = 100,
                    tabu_tenure      = 7,
                    max_iter_ts      = 50,
                    stagnation_limit = 10
                )
                obj_igts = compute_objectives(seq_igts, pt, due_dates, weights)
                results[name]['ig_ts'][obj].append(obj_igts[obj])

    return results


def plot_comparison(results):
    """
    Affiche et sauvegarde les courbes de comparaison.
    """
    os.makedirs("resultats/plots", exist_ok=True)

    objectives = ['TT', 'TWT', 'T_max', 'NT']
    obj_labels = {
        'TT':    'Total Tardiness',
        'TWT':   'Total Weighted Tardiness',
        'T_max': 'Maximum Tardiness',
        'NT':    'Number of Tardy Jobs'
    }

    methods = {
        'taillard': {'label': 'Taillard (identity)', 'color': '#607D8B', 'marker': 'o'},
        'nehedd':   {'label': 'NEHedd',               'color': '#2196F3', 'marker': 's'},
        'ig':       {'label': 'IG',                   'color': '#4CAF50', 'marker': '^'},
        'ig_ts':    {'label': 'IG-TS',                'color': '#E91E63', 'marker': 'D'}  # ← nouveau
    }

    for name, data in results.items():
        n_instances = len(data['taillard']['TT'])
        x           = list(range(1, n_instances + 1))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Comparaison des méthodes — {name}",
            fontsize   = 14,
            fontweight = 'bold'
        )

        axes_flat = axes.flatten()

        for ax, obj in zip(axes_flat, objectives):
            for method, style in methods.items():
                ax.plot(
                    x,
                    data[method][obj],
                    label      = style['label'],
                    color      = style['color'],
                    marker     = style['marker'],
                    linewidth  = 2,
                    markersize = 6
                )

            ax.set_title(obj_labels[obj], fontsize=11)
            ax.set_xlabel("Instance")
            ax.set_ylabel("Valeur")
            ax.set_xticks(x)
            ax.legend(fontsize=9)
            ax.grid(linestyle='--', alpha=0.4)

        plt.tight_layout()
        filepath = f"resultats/plots/{name}_comparison.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot sauvegardé : {filepath}")


def run_plots(datasets):
    """
    Pipeline complet : calcul + affichage.
    """
    print("\n" + "="*50)
    print("Calcul des résultats...")
    results = compute_all_results(datasets)

    print("\n" + "="*50)
    print("Génération des plots...")
    plot_comparison(results)
