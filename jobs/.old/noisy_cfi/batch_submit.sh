# #!/bin/bash
# Batch submit jobs to Graham

FOLDER="noisy_cfi_batch_sweep"
ANSATZ="cnot_2local_dephased_ansatz"
for n in {1..8};
  do
  for k in {1..8};
    do
      JOB_NAME="noisy_cfi_n${n}_k${k}_${ANSATZ}"
      echo $JOB_NAME
      time=$((2 * n))
      #echo "0:${n}0:00"
      sbatch --output="slurm_${JOB_NAME}.out" --time="0:${time}0:00" --job-name $JOB_NAME --export=NQUBIT=$n,FOLDER=${FOLDER},KLAYER=$k,ANSATZ=${ANSATZ} submit.sh
  done
done
sleep 0.5s
