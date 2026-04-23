import os
import sys
import csv
import traceback
from src.data_loader import load_instance
from src.math_model import solve_milp_tt

def save_fail_result(result_file, subdir, instance_file, status_msg):
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    rows = [
        ["Job", "Due date", "Start M1", "Completion Cj", "Tardiness", "Tardy"],
        ["", "", "", "", "", ""],
        ["Status", "TT", "Temps (s)", "Gap (%)", "BestBound", "Objective"],
        [status_msg, "", "", "", "", ""]
    ]

    with open(result_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

def append_summary(summary_file, subdir, instance_file, result):
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    file_exists = os.path.isfile(summary_file)

    with open(summary_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Dataset", "Instance", "Status", "TT", "Temps (s)",
                "Gap (%)", "BestBound", "Objective", "Sequence"
            ])

        writer.writerow([
            subdir,
            instance_file,
            result.get("status", ""),
            result.get("TT", ""),
            result.get("time", ""),
            result.get("gap", ""),
            result.get("best_bound", ""),
            result.get("objective_value", ""),
            " ".join(str(j + 1) for j in result.get("sequence", [])) if result.get("sequence") else ""
        ])

        f.flush()
        os.fsync(f.fileno())

def run_instance(instance_id, subdir):
    subdir = "20j_5m"
    instance_file = f"instance_{instance_id}.csv"

    print(f"{'='*60}", flush=True)
    print(f"Dataset  : {subdir}", flush=True)
    print(f"Instance : {instance_file}", flush=True)
    print(f"{'='*60}", flush=True)

    results_dir_milp = "resultats/milp_new"
    os.makedirs(os.path.join(results_dir_milp, subdir), exist_ok=True)

    instance_path = os.path.join("data/instances", subdir, instance_file)
    result_file = os.path.join(results_dir_milp, subdir, f"result_{instance_id}.csv")
    summary_file = os.path.join(results_dir_milp, subdir, "summary_results.csv")

    try:
        print(f"[LOAD] Chargement de {instance_path}", flush=True)
        instance = load_instance(instance_path)

        pt = instance["processing_times"]
        due_dates = instance["due_date"]

        print("[RUN] Résolution MILP...", flush=True)
        result = solve_milp_tt(
            pt,
            due_dates,
            time_limit= 24 * 3600,
            filepath=result_file,
            instance_name=f"{subdir}/{instance_file}"
        )

        if result is None:
            result = {
                "status": "NO_SOLUTION",
                "TT": "",
                "time": "",
                "gap": "",
                "best_bound": "",
                "objective_value": "",
                "sequence": []
            }
            save_fail_result(result_file, subdir, instance_file, "NO_SOLUTION")

        append_summary(summary_file, subdir, instance_file, result)

        print(f"\n{'='*60}", flush=True)
        print("Résultat final :", flush=True)
        print(f"  Status      : {result['status']}", flush=True)
        print(f"  TT          : {result['TT']}", flush=True)
        print(f"  Temps (s)   : {result['time']}", flush=True)
        print(f"  Gap (%)     : {result['gap']}", flush=True)
        print(f"  Séquence    : {[j+1 for j in result['sequence']] if result['sequence'] else []}", flush=True)
        print(f"  Fichier CSV : {result_file}", flush=True)
        print(f"  Summary CSV : {summary_file}", flush=True)

    except Exception as e:
        err_msg = f"FAILED: {type(e).__name__}: {e}"
        print(err_msg, flush=True)
        traceback.print_exc()

        save_fail_result(result_file, subdir, instance_file, err_msg)

        fail_result = {
            "status": err_msg,
            "TT": "",
            "time": "",
            "gap": "",
            "best_bound": "",
            "objective_value": "",
            "sequence": []
        }
        append_summary(summary_file, subdir, instance_file, fail_result)

        sys.exit(1)

if __name__ == "__main__":
    instance_id = int(sys.argv[1])
    subdir = sys.argv[2]
    run_instance(instance_id, subdir)