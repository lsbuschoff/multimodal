#!/bin/bash

#SBATCH -J multi
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH -p gpu_p
#SBATCH --qos gpu_short
#SBATCH --constraint=a100_80gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 5:00:00
#SBATCH --nice=10000
#SBATCH --cpus-per-task=10

source $HOME/.bashrc
conda init
conda activate fuyu
python3 main_fuyu.py --dataset "MICHOTTE"
