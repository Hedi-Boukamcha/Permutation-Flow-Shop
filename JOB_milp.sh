#!/bin/bash
#SBATCH --account=def-adhaj
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --array=1-20               # 20 instances en parallèle (10 par taille)
#SBATCH --output=logs/%A_%a_out.txt
#SBATCH --error=logs/%A_%a_err.txt

module load python/3.11
source ~/env_pfsp/bin/activate

cd ~/projet

# Créer les répertoires nécessaires s'ils n'existent pas
mkdir -p logs
mkdir -p results/milpF/20j_5m
mkdir -p results/milpF/50j_10m


# Mapper index → dataset + instance
if [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
    SUBDIR=20j_5m
    INSTANCE_ID=$SLURM_ARRAY_TASK_ID
else
    SUBDIR=50j_10m
    INSTANCE_ID=$((SLURM_ARRAY_TASK_ID - 10))
fi

INSTANCE_FILE=instance_${INSTANCE_ID}.csv

if [ -z "$SUBDIR" ] || [ -z "$INSTANCE_FILE" ]; then
    echo "Erreur: SUBDIR ou INSTANCE_FILE vide"
    exit 1
fi

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SUBDIR=$SUBDIR"
echo "INSTANCE_FILE=$INSTANCE_FILE"

python -u CLUSTER.py "$SUBDIR" "$INSTANCE_FILE"