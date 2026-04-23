#!/bin/bash
#SBATCH --account=def-adhaj
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --array=1-10%5               # ← 10 jobs en parallèle pour chaque ensemble, 5 à la fois
#SBATCH --output=logs/%A_%a_%j_out.txt
#SBATCH --error=logs/%A_%a_%j_err.txt

module load python/3.11
source ~/env_pfsp/bin/activate

cd ~/projet

# Créer les répertoires nécessaires s'ils n'existent pas
mkdir -p logs
mkdir -p results/milpF/20j_5m
mkdir -p results/milpF/50j_10m
mkdir -p data/instances/20j_5m
mkdir -p data/instances/50j_10m

# Soumettre les jobs pour les instances 20j_5m
job_id_20j_5m=$(sbatch --parsable --export=subdir="20j_5m" job_instance.sh)
echo "Job soumis pour 20j_5m : $job_id_20j_5m"

# Soumettre les jobs pour les instances 50j_10m
job_id_50j_10m=$(sbatch --parsable --export=subdir="50j_10m" job_instance.sh)
echo "Job soumis pour 50j_10m : $job_id_50j_10m

#!/bin/bash

# Arguments : instance_id
python TT_cluster.py $SLURM_ARRAY_TASK_ID $subdir