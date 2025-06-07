#!/bin/sh

#SBATCH --job-name=ML-Graph
#SBATCH --partition=gpu-a100-small
#SBATCH --account=research-eemcs-diam
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G

module load 2023r1
module load openmpi
module load python/3.11
module load py-pip

srun pip install -r requirements.txt
srun python main.py > output.log