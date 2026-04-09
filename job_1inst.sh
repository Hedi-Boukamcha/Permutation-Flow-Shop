#!/bin/bash
#SBATCH --job-name=pfsp_TT_milp
#SBATCH --account=def-adhaj
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/%j_out.txt
#SBATCH --error=logs/%j_err.txt

module load python/3.11
source ~/env_pfsp/bin/activate

cd ~/projet
mkdir -p logs
mkdir -p resultats/milp_new/20j_5m

# Instance ID passé en argument
# Exemple : sbatch jobTT.sh 0  → instance 1
python TT_cluster.py ${1:-0}