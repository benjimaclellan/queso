# #!/bin/bash
# Starts an interactive job on Graham with one GPU
srun  --time=03:00:00 --mem=3000M --gres=gpu:1 --pty /bin/bash

module load python/3.9 scipy-stack
source ~/bash-profiles/queso
source ~/venv/queso_venv/bin/activate
module load python/3.9 scipy-stack
