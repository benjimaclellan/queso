# #!/bin/bash
# Batch submit jobs to Graham

FOLDER="pure_qfi_batch_sweep"
ANSATZ="cnot_2local_ansatz"

for n in 2;
  do
  for k in 2;
    do
      JOB_NAME="n${n}_k${k}_${ANSATZ}"
      echo $JOB_NAME
      #echo "0:${n}0:00"
      sbatch --output="slurm_${JOB_NAME}.out" --time="00:${n}0:00" --job-name $JOB_NAME --export=NQUBIT=$n,FOLDER=${FOLDER},KLAYER=$k,ANSATZ=${ANSATZ} submit.sh
  done
done
sleep 0.5s
