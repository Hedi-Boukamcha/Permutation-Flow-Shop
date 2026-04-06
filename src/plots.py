import random
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import os
from src.scheduler import compute_completion_times



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


def plot_gantt(sequence, processing_times, due_dates, weights=None,
               title="Gantt Chart", filename="gantt.png"):

    n_machines = processing_times.shape[0]
    n_jobs     = len(sequence)

    output_dir  = os.path.join("data", "gantt")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    C      = compute_completion_times(sequence, processing_times)
    #colors = plt.cm.tab20(np.linspace(0, 1, n_jobs))
    colors = generate_colors(n_jobs, seed=42)

    # Calcul des start times
    starts = np.zeros((n_machines, n_jobs), dtype=int)
    for j in range(n_jobs):
        for i in range(n_machines):
            if j == 0 and i == 0:
                starts[i][j] = 0
            elif j == 0:
                starts[i][j] = C[i-1][j]
            elif i == 0:
                starts[i][j] = C[i][j-1]
            else:
                starts[i][j] = max(C[i-1][j], C[i][j-1])

    # Completion times de la dernière machine
    completion_times = C[n_machines-1]  # shape: (n_jobs,)

    fig, (ax_gantt, ax_table) = plt.subplots(
        2, 1,
        figsize     = (16, n_machines * 1.5 + n_jobs * 0.45 + 2),
        gridspec_kw = {'height_ratios': [n_machines * 1.5, n_jobs * 0.45]}
    )

    # ─── Gantt ───────────────────────────────────────────────
    for j, job in enumerate(sequence):
        for i in range(n_machines):
            start    = starts[i][j]
            end      = C[i][j]
            duration = end - start

            ax_gantt.barh(
                y         = i,
                width     = duration,
                left      = start,
                color     = colors[j],
                edgecolor = 'white',
                linewidth = 0.8,
                height    = 0.6
            )

            # Label job dans le bloc
            if duration > 10:
                ax_gantt.text(
                    x          = start + duration / 2,
                    y          = i,
                    s          = f"J{job+1}",
                    ha         = 'center',
                    va         = 'center',
                    fontsize   = 7,
                    color      = 'white',
                    fontweight = 'bold'
                )

    # Marquer la date de completion de chaque job sur l'axe x
    for j, job in enumerate(sequence):
        cj = completion_times[j]

        # Ligne verticale à la completion time
        ax_gantt.axvline(
            x         = cj,
            color     = colors[j],
            linestyle = ':',
            linewidth = 1,
            alpha     = 0.8
        )

        # Label C_j sur l'axe x
        ax_gantt.text(
            x        = cj,
            y        = n_machines + 0.1,
            s        = f"C{job+1}={cj}",
            fontsize = 6,
            color    = colors[j],
            ha       = 'center',
            va       = 'bottom',
            rotation = 90
        )

    # Axes
    ax_gantt.set_yticks(range(n_machines))
    ax_gantt.set_yticklabels([f"Machine {i+1}" for i in range(n_machines)])
    ax_gantt.invert_yaxis()  # Machine 1 en haut
    ax_gantt.set_xlabel("Time")
    ax_gantt.set_title(title)
    ax_gantt.grid(axis='x', linestyle='--', alpha=0.3)

    # Supprimer les ticks x par défaut
    ax_gantt.set_xticks([])


    # ─── Tableau ─────────────────────────────────────────────
    ax_table.axis('off')

    col_labels = ["Job", "Due date", "Weight",
                  "Start M1", "Completion Cj", "Tardiness", "Tardy"]
    table_data = []

    for j, job in enumerate(sequence):
        start_m1  = starts[0][j]
        cj        = completion_times[j]
        dd        = due_dates[job]
        w         = weights[job] if weights is not None else 1
        tardiness = max(0, cj - dd)
        tardy     = "YES" if tardiness > 0 else "NO"

        table_data.append([
            f"J{job+1}",
            dd,
            w,
            start_m1,
            cj,
            tardiness,
            tardy
        ])

    table = ax_table.table(
        cellText  = table_data,
        colLabels = col_labels,
        cellLoc   = 'center',
        loc       = 'center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.2)

    # Header
    for k in range(len(col_labels)):
        table[0, k].set_facecolor('#DDDDDD')
        table[0, k].set_text_props(fontweight='bold')

    # Couleur selon tardy
    for j in range(len(table_data)):
        color_row = '#FFCCCC' if table_data[j][6] == "YES" else '#CCFFCC'
        for k in range(len(col_labels)):
            table[j+1, k].set_facecolor(color_row)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Gantt sauvegardé : {output_path}")