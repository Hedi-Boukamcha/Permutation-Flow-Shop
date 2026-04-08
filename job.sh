#!/bin/bash
#SBATCH --job-name=pfsp_milp
#SBATCH --account=def-nomduprof    # ← remplacer par le compte du prof
#SBATCH --time=24:00:00            # 2 heures par instance
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19               # 10 instances en parallèle
#SBATCH --output=logs/%A_%a_out.txt
#SBATCH --error=logs/%A_%a_err.txt

# Charger l'environnement
module load python/3.11
source ~/env_pfsp/bin/activate

# Aller dans le dossier projet
cd ~/projet

# Créer les dossiers nécessaires
mkdir -p logs
mkdir -p resultats/milp/20j_5m
mkdir -p resultats/milp/50j_10m

# Mapper index → dataset + instance
if [ $SLURM_ARRAY_TASK_ID -lt 10 ]; then
    python milp_cluster.py $SLURM_ARRAY_TASK_ID tai20j_5m
else
    INSTANCE_ID=$((SLURM_ARRAY_TASK_ID - 10))
    python milp_cluster.py $INSTANCE_ID tai50j_10m
fi