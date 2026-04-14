#!/bin/bash
#SBATCH --job-name=pfsp_milp
#SBATCH --account=def-adhaj    # ← remplacer par le compte du prof
#SBATCH --time=4-00:00:00          # 4 jours par instance
#SBATCH --mem=200G                  # 64 Go de mémoire RAM par tâche
#SBATCH --cpus-per-task=4
#SBATCH --array=1-20               # 20 instances en parallèle (10 par taille)
#SBATCH --output=logs/%A_%a_out.txt
#SBATCH --error=logs/%A_%a_err.txt

# Charger l'environnement
module load python/3.11
source ~/env_pfsp/bin/activate

# Aller dans le dossier projet
cd ~/projet

# Créer les dossiers nécessaires
mkdir -p logs
mkdir -p results/milp_tt/20j_5m
mkdir -p results/milp_tt/50j_10m

# Mapper index → dataset + instance
if [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
    SUBDIR=20j_5m
    INSTANCE_ID=$SLURM_ARRAY_TASK_ID
else
    SUBDIR=50j_10m
    INSTANCE_ID=$((SLURM_ARRAY_TASK_ID - 10))
fi

INSTANCE_FILE=instance_${INSTANCE_ID}.csv

python milp_cluster.py $SUBDIR $INSTANCE_FILE
