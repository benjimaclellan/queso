# #!/bin/bash
# Starts an interactive job on Graham with one GPU
srun  --time=03:00:00 --mem=3000M --gres=gpu:1 --account=def-rgmelko --pty /bin/bash

srun  --time=03:00:00 --mem=3000M --account=def-rgmelko --pty /bin/bash

salloc  --time=03:00:00 --mem=3000M --gres=gpu:1


source ~/bash-profiles/queso
source ~/venv/queso_venv/bin/activate
module load cuda/11.4
module load cudnn/8.2.0
module load python/3.9 scipy-stack

# to log into disconnected session, just ssh into node. No password needed
# ssh node-name  (e.g., ssh gra977)