import time
import csv
import os
import numpy as np

from src.data_loader      import load_all
from src.dd_generator     import generate_due_dates_brah, generate_weights
from src.scheduler        import compute_objectives
from src.initial_solution import nehedd
from src.IG_TS_approche   import ig, ig_ts
from config               import FOLDE_MAP


def run_aggregated_tt(datasets):
    """
    Génère un tableau agrégé pour l'objectif TT uniquement.

    Colonnes :
    instance | obj MILP | status MILP | ct MILP |
    obj NEHedd | dev NEHedd | ct NEHedd |
    obj IG | dev IG | ct IG |
    obj IG-TS | dev IG-TS | ct IG-TS
    """
    os.makedirs("resultats/aggregated", exist_ok=True)

    for name, instances in datasets.items():
        folder = FOLDE_MAP.get(name, name)
        rows   = []

        print(f"\n{'='*50}")
        print(f"Dataset : {name} — Objectif : TT")

        for idx, inst in enumerate(instances):
            pt          = inst['processing_times']
            due_dates   = generate_due_dates_brah(inst, tau=2)
            weights     = generate_weights(inst)
            instance_id = idx + 1

            print(f"\n  Instance {instance_id}...")
            row = {'instance': instance_id}

            # ── 1. MILP — lire depuis CSV Narval ─────────────
            milp_path = f"resultats/milp/{folder}/instance_{instance_id}.csv"
            if os.path.exists(milp_path):
                with open(milp_path) as f:
                    for line in f.readlines():
                        if 'FEASIBLE' in line or 'OPTIMAL' in line:
                            parts = line.strip().split(',')
                            row['status MILP'] = parts[0]
                            row['obj MILP']    = int(parts[1])
                            row['ct MILP']     = 'N/A'
            else:
                row['obj MILP']    = 'N/A'
                row['status MILP'] = 'N/A'
                row['ct MILP']     = 'N/A'

            ref = row['obj MILP']

            # ── 2. NEHedd ────────────────────────────────────
            t0      = time.time()
            seq     = nehedd(pt, due_dates, weights, objective='TT')
            obj     = compute_objectives(seq, pt, due_dates, weights)
            ct      = round(time.time() - t0, 4)
            val_neh = obj['TT']

            row['obj NEHedd'] = val_neh
            row['dev NEHedd'] = round((val_neh - ref) / ref * 100, 2) if ref != 'N/A' else 'N/A'
            row['ct NEHedd']  = ct

            # ── 3. IG ────────────────────────────────────────
            t0           = time.time()
            seq, _, _    = ig(
                processing_times = pt,
                due_dates        = due_dates,
                weights          = weights,
                objective        = 'TT',
                k                = 4,
                max_iter         = 100
            )
            obj    = compute_objectives(seq, pt, due_dates, weights)
            ct     = round(time.time() - t0, 4)
            val_ig = obj['TT']

            row['obj IG'] = val_ig
            row['dev IG'] = round((val_ig - ref) / ref * 100, 2) if ref != 'N/A' else 'N/A'
            row['ct IG']  = ct

            # ── 4. IG-TS ─────────────────────────────────────
            t0           = time.time()
            seq, _, _    = ig_ts(
                processing_times = pt,
                due_dates        = due_dates,
                weights          = weights,
                objective        = 'TT',
                k                = 4,
                max_iter         = 100,
                tabu_tenure      = 7,
                max_iter_ts      = 50,
                stagnation_limit = 10
            )
            obj       = compute_objectives(seq, pt, due_dates, weights)
            ct        = round(time.time() - t0, 4)
            val_igts  = obj['TT']

            row['obj IG-TS'] = val_igts
            row['dev IG-TS'] = round((val_igts - ref) / ref * 100, 2) if ref != 'N/A' else 'N/A'
            row['ct IG-TS']  = ct

            rows.append(row)
            print(f"    MILP={ref} | NEHedd={val_neh} | IG={val_ig} | IG-TS={val_igts}")

        # ── Sauvegarde ────────────────────────────────────────
        filepath = f"resultats/aggregated/{folder}_TT_summary.csv"
        _save_csv(rows, filepath)
        print(f"\n {filepath}")


def _save_csv(rows, filepath):

    fieldnames = [
        'instance',
        'obj MILP',   'status MILP', 'ct MILP',
        'obj NEHedd', 'dev NEHedd',  'ct NEHedd',
        'obj IG',     'dev IG',      'ct IG',
        'obj IG-TS',  'dev IG-TS',   'ct IG-TS',
    ]

    with open(filepath, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

        # ── Ligne moyenne ─────────────────────────────────────
        f.write('\n')
        avg_row  = {'instance': 'AVERAGE'}
        num_cols = [
            'obj MILP',
            'obj NEHedd', 'dev NEHedd', 'ct NEHedd',
            'obj IG',     'dev IG',     'ct IG',
            'obj IG-TS',  'dev IG-TS',  'ct IG-TS'
        ]

        for col in num_cols:
            vals = [r[col] for r in rows if r.get(col) != 'N/A']
            avg_row[col] = round(np.mean(vals), 4) if vals else 'N/A'

        writer.writerow(avg_row)