#!/bin/bash
#SBATCH --mail-user=bmaclell@uwaterloo.ca
#SBATCH --mail-type=FAIL

FOLDER=$1

module purge
module load python/3.9 scipy-stack
module load cuda/11.4
module load cudnn/8.2.0
source ~/bash-profiles/queso
source ~/venv/queso_venv/bin/activate

echo "${FOLDER}"
cd ~/projects/def-rgmelko/bmaclell/queso
python queso/sample_circuit.py --folder "${FOLDER}"
python queso/sample_circuit_testing.py --folder "${FOLDER}"
