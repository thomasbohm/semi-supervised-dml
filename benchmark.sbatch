#!/bin/bash
#SBATCH --job-name="DML"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/boehmtho/benchmarks/%j.out
#SBATCH --error=/storage/slurm/boehmtho/benchmarks/%j.out

source ./dml_env/bin/activate
python3 benchmarker.py \
--experiment_name test4 \
--dataset~OVERRIDE~ {CUB200: {}} \
--models {trunk~OVERRIDE~: {resnet50: {pretrained: imagenet}}, embedder~OVERRIDE~: {MLP: {layer_sizes: [512]}}} \
--num_epochs_train 2