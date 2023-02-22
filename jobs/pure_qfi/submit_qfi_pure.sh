#!/bin/bash
#SBATCH --mem=32G
#SBATCH --account=def-rgmelko
#SBATCH --nodes=1
##SBATCH --gpus-per-node=1
#SBATCH --time=0:10:00
#SBATCH --mail-user=bmaclell@uwaterloo.ca
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_%J.out

module purge
module load python/3.9 scipy-stack
module load cuda/11.4
module load cudnn/8.2.0
source ~/.bash_profile
source ~/queso_venv/bin/activate

echo "${FOLDER}" ${NQUBIT} ${KLAYER} ${ANSATZ}

cd ~/projects/def-rgmelko/bmaclell/queso
python experiments/optimize_pure_qfi.py --folder "${FOLDER}" --n ${NQUBIT} --k ${KLAYER} --ansatz ${ANSATZ} -seed 0
