# #!/bin/bash
# Batch submit jobs to Graham

FOLDER="pure_qfi_batch_sweep"
ANSATZ="cnot_2local_ansatz"

for n in {9..12};
  do
  for k in 1 2 3 4 5 6 7 8 9 10 11 12;
    do
      JOB_NAME="pure_qfi_n${n}_k${k}_${ANSATZ}"
      echo $JOB_NAME
      #echo "0:${n}0:00"
      time=$((2 * n))
      sbatch --output="slurm_${JOB_NAME}.out" --time="00:${time}0:00" --job-name $JOB_NAME --export=NQUBIT=$n,FOLDER=${FOLDER},KLAYER=$k,ANSATZ=${ANSATZ} submit.sh
  done
done
sleep 0.5s
