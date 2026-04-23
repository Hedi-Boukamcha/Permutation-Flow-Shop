#!/bin/bash
#SBATCH --account=def-adhaj
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/submit_all_%j.out
#SBATCH --error=logs/submit_all_%j.err

mkdir -p logs
mkdir -p jobs

DATASETS=("20j_5m" "50j_10m")

for subdir in "${DATASETS[@]}"; do
    for instance_path in data/instances/${subdir}/*.csv; do
        instance_file=$(basename "$instance_path")
        job_name="milp_${subdir}_${instance_file%.csv}"

        cat > jobs/${job_name}.sh <<EOF
#!/bin/bash
#SBATCH --account=def-adhaj
#SBATCH --job-name=${job_name}
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err

module load python/3.10

cd \$SLURM_SUBMIT_DIR

python run_one_milp_instance.py ${subdir} ${instance_file}
EOF

        sbatch jobs/${job_name}.sh
    done
done