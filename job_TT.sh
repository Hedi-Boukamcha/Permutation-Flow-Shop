#!/bin/bash
#SBATCH --job-name=pfsp_TT_milp
#SBATCH --account=def-adhaj
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9                  # ← 10 jobs en parallèle
#SBATCH --output=logs/%A_%a_out.txt
#SBATCH --error=logs/%A_%a_err.txt

module load python/3.11
source ~/env_pfsp/bin/activate

cd ~/projet
mkdir -p logs
mkdir -p resultats/milp_new/20j_5m


# Arguments : dataset method time_limit output
python TT_cluster.py $SLURM_ARRAY_TASK_ID