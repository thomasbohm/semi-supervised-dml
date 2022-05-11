#!/bin/bash
#SBATCH --job-name="DML"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --time=3-00:00:00
#SBATCH --output=/storage/slurm/boehmtho/cub/hyper-10-%j.out
#SBATCH --error=/storage/slurm/boehmtho/cub/hyper-10-%j.out
source ./dml_env/bin/activate
python3 main.py --config_path="config/cub.yaml"