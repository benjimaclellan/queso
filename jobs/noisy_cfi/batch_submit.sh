# #!/bin/bash
# Batch submit jobs to Graham

FOLDER="noisy_cfi_batch_sweep"
ANSATZ="cnot_2local_dephased_ansatz"
for n in 8;
  do
  for k in 2 3 4 5 6;
    do
      JOB_NAME="n${n}_k${k}_${ANSATZ}"
      echo $JOB_NAME
      #echo "0:${n}0:00"
      sbatch --output="slurm_${JOB_NAME}.out" --time="26:00:00" --job-name $JOB_NAME --export=NQUBIT=$n,FOLDER=${FOLDER},KLAYER=$k,ANSATZ=${ANSATZ} submit.sh
  done
done
sleep 0.5s
