# #!/bin/bash
# Run a batch of jobs on a local laptop

FOLDER="pure_qfi_barren_plateau"

for n in 1 2 3 4 5 6 7 8;
  do
  for k in 1 2 3 4 5 6 7 8;
    do
      JOB_NAME="Sample circuit for barren plateaus: n = ${n}, k = ${k}"
      echo $JOB_NAME
      python experiments/barren_plateaus.py --folder "${FOLDER}" --n ${n} --k ${k}
  done
done
sleep 0.5s
