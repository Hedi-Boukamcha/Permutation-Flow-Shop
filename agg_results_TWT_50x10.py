import os
import pandas as pd


# ─────────────────────────────────────────────
# À MODIFIER À LA MAIN
# ─────────────────────────────────────────────
SUMMARY_FILES = {
    "HEUR": "results/my_heuristic/50j_10m/summary_heuristic_TWT.csv",
    "TMIG": "results/tm_ig/50j_10m/summary_TWT.csv",
    "NEHEDD_IT1": "results/nehedd_it1/50j_10m/summary_NEHedd_IT1_TWT.csv",
}

OUTPUT_CSV = "results/aggregated/50j_10m/aggregated_TWT.csv"
OUTPUT_TEX = "results/aggregated/50j_10m/aggregated_TWT.tex"


def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def aggregate_tt():
    final_df = None

    for heur_name, file_path in SUMMARY_FILES.items():
        print(f"[READ] {heur_name} -> {file_path}")
        if not os.path.exists(file_path):
            print(f"[WARNING] fichier introuvable : {file_path}")
            continue

        df = pd.read_csv(file_path)

        instance_col = find_col(df, [
            "Instance", "instance", "Instance file",
            "instance_file", "File", "file"
        ])

        tt_col = find_col(df, [
            "TWT", "Objective", "objective", "Value", "value"
        ])

        time_col = find_col(df, [
            "time", "Time", "Temps (s)", "Temps",
            "Runtime (s)", "runtime"
        ])

        if instance_col is None:
            raise ValueError(f"Colonne instance introuvable dans {file_path}")

        if tt_col is None:
            raise ValueError(f"Colonne TT introuvable dans {file_path}")

        if time_col is None:
            raise ValueError(f"Colonne temps introuvable dans {file_path}")

        tmp = df[[instance_col, tt_col, time_col]].copy()

        tmp = tmp.rename(columns={
            instance_col: "Instance",
            tt_col: f"{heur_name}_TT",
            time_col: f"{heur_name}_time"
        })
        tmp["Instance"] = tmp["Instance"].astype(str).str.extract(r'(\d+)').astype(int)

        if final_df is None:
            final_df = tmp
        else:
            final_df = pd.merge(final_df, tmp, on="Instance", how="outer")

    if final_df is None:
        raise ValueError("Aucun fichier valide trouvé.")

    final_df = final_df.sort_values(by="Instance")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[SAVE] CSV créé : {OUTPUT_CSV}")
    return final_df


def export_latex(df):
    os.makedirs(os.path.dirname(OUTPUT_TEX), exist_ok=True)

    heuristics = list(SUMMARY_FILES.keys())

    # Arrondir les colonnes numériques
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # Header LaTeX
    col_format = "l|" + "|".join(["cc" for _ in heuristics])

    header = f"""
\\begin{{table}}[H]
\\centering
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_format}}}
\\toprule
"""

    header += "\\textbf{Instance} "

    for heur in heuristics:
        header += f"& \\multicolumn{{2}}{{c}}{{\\textbf{{{heur}}}}} "

    header += "\\\\\n"

    header += " "
    for heur in heuristics:
        header += "& TT & Time(s) "

    header += "\\\\\n\\midrule\n"

    rows = ""

    for _, row in df.iterrows():
        line = f"{row['Instance']}"

        for heur in heuristics:
            tt_col = f"{heur}_TT"
            time_col = f"{heur}_time"

            tt_val = row[tt_col] if tt_col in df.columns else "-"
            time_val = row[time_col] if time_col in df.columns else "-"

            line += f" & {tt_val} & {time_val}"

        rows += line + " \\\\\n"

    # Résumé : moyenne TT + moyenne temps
    summary = "\\midrule\n"

    summary += "\\textbf{Avg TT}"
    for heur in heuristics:
        tt_col = f"{heur}_TT"
        if tt_col in df.columns:
            summary += f" & {df[tt_col].mean():.2f} & -"
        else:
            summary += " & - & -"
    summary += " \\\\\n"

    summary += "\\textbf{Avg Time}"
    for heur in heuristics:
        time_col = f"{heur}_time"
        if time_col in df.columns:
            summary += f" & - & {df[time_col].mean():.2f}"
        else:
            summary += " & - & -"
    summary += " \\\\\n"

    footer = r"""
\bottomrule
\end{tabular}
}
\caption{Résultats agrégés des heuristiques pour l'objectif TT}
\label{tab:aggregated_tt}
\end{table}
"""

    full_latex = header + rows + summary + footer

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(full_latex)

    print(f"[SAVE] LaTeX créé : {OUTPUT_TEX}")


def main():
    df = aggregate_tt()
    export_latex(df)


if __name__ == "__main__":
    main()