import os
import pandas as pd


BASE_PATHS = {
    "HEUR": "results/my_heuristic/20j_5m/summary_heuristic_TT.csv",
    "TMIG": "results/tm_ig/20j_5m/summary_TT.csv",
    "NEHEDD_IT1": "results/nehedd_it1/20j_5m/summary_NEHedd_IT1_TT.csv",
}

OBJECTIVES = ["TT", "TWT", "T_max", "NT"]
SUBDIRS = ["20j_5m", "50j_10m"]

OUTPUT_ROOT = "results/aggregated"


def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def aggregate_one(subdir, obj):
    final_df = None

    for heur_name, base_path in BASE_PATHS.items():
        file_path = os.path.join(base_path, subdir, f"summary_{obj}.csv")

        if not os.path.exists(file_path):
            print(f"[WARNING] introuvable : {file_path}")
            continue

        df = pd.read_csv(file_path)

        instance_col = find_col(df, ["Instance", "instance", "Instance file", "instance_file"])
        obj_col = find_col(df, [obj, "Objective", "objective", "Value", "value"])
        time_col = find_col(df, ["time", "Time", "Temps (s)", "Runtime (s)", "runtime"])

        if instance_col is None or obj_col is None or time_col is None:
            print(f"[SKIP] colonnes manquantes : {file_path}")
            continue

        tmp = df[[instance_col, obj_col, time_col]].copy()

        tmp = tmp.rename(columns={
            instance_col: "Instance",
            obj_col: f"{heur_name}_{obj}",
            time_col: f"{heur_name}_time"
        })

        tmp["Instance"] = (
            tmp["Instance"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .astype(int)
        )

        if final_df is None:
            final_df = tmp
        else:
            final_df = pd.merge(final_df, tmp, on="Instance", how="outer")

    if final_df is None:
        print(f"[ERROR] aucun résultat pour {subdir} - {obj}")
        return

    final_df = final_df.sort_values("Instance")

    ordered_cols = ["Instance"]
    for heur in BASE_PATHS.keys():
        ordered_cols += [f"{heur}_{obj}", f"{heur}_time"]

    final_df = final_df[[c for c in ordered_cols if c in final_df.columns]]

    output_dir = os.path.join(OUTPUT_ROOT, subdir)
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"aggregated_{obj}.csv")
    output_tex = os.path.join(output_dir, f"aggregated_{obj}.tex")

    final_df.to_csv(output_csv, index=False)
    export_latex(final_df, obj, output_tex)

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_tex}")


def export_latex(df, obj, output_tex):
    heuristics = list(BASE_PATHS.keys())

    col_format = "l|" + "|".join(["cc" for _ in heuristics])

    header = f"""
\\begin{{table}}[H]
\\centering
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_format}}}
\\toprule
\\textbf{{Instance}}
"""

    for heur in heuristics:
        header += f"& \\multicolumn{{2}}{{c}}{{\\textbf{{{heur}}}}} "

    header += "\\\\\n"

    for heur in heuristics:
        header += f"& {obj} & Time(s) "

    header += "\\\\\n\\midrule\n"

    rows = ""
    for _, row in df.iterrows():
        line = f"{int(row['Instance'])}"

        for heur in heuristics:
            obj_col = f"{heur}_{obj}"
            time_col = f"{heur}_time"

            obj_val = row[obj_col] if obj_col in df.columns else "-"
            time_val = row[time_col] if time_col in df.columns else "-"

            line += f" & {obj_val} & {time_val}"

        rows += line + " \\\\\n"

    summary = "\\midrule\n"

    summary += f"\\textbf{{Avg {obj}}}"
    for heur in heuristics:
        obj_col = f"{heur}_{obj}"
        if obj_col in df.columns:
            summary += f" & {df[obj_col].mean():.2f} & -"
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

    footer = f"""
\\bottomrule
\\end{{tabular}}
}}
\\caption{{Résultats agrégés pour l'objectif {obj}}}
\\label{{tab:aggregated_{obj}}}
\\end{{table}}
"""

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(header + rows + summary + footer)


def main():
    for subdir in SUBDIRS:
        for obj in OBJECTIVES:
            print(f"[RUN] {subdir} - {obj}")
            aggregate_one(subdir, obj)


if __name__ == "__main__":
    main()