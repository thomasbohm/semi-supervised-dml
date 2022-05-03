#!/bin/bash
#SBATCH --job-name="DML"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/boehmtho/sop-100-%j.out
#SBATCH --error=/storage/slurm/boehmtho/sop-100-%j.out
cd ~/semi-supervised-dml/
source ~/semi-supervised-dml/dml_env/bin/activate
srun python3 main.py --config_path="config/sop.yaml"