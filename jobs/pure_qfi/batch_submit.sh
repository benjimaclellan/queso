# #!/bin/bash
# Batch submit jobs to Graham

FOLDER="pure_qfi_batch_sweep"
ANSATZ="cnot_2local_ansatz"

for n in 1;
  do
  for k in 1 2 3 4 5 6 7 8 9 10 11 12;
    do
      JOB_NAME="n${n}_k${k}"
      echo $JOB_NAME
      #echo "0:${n}0:00"
      sbatch --output="slurm_${JOB_NAME}.out" --time="00:${n}0:00" --job-name $JOB_NAME --export=NQUBIT=$n,FOLDER=${FOLDER},KLAYER=$k,ANSATZ=${ANSATZ} submit_qfi_pure.sh
  done
done
sleep 0.5s
