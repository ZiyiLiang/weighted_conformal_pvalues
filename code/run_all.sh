#!/bin/bash

#DATA_LIST=(0 1 2 3 4 5 6)

DATA_LIST=(0 1 2 3 5 6)

#SEED_LIST=(0)

# Slurm parameters
MEMO=16G                             # Memory required (16GB)
TIME=14-00:00:00                      # Time required (14 days)
CORE=1                               # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

LOGS="logs"
mkdir -p $LOGS

OUT_DIR="results/"
mkdir -p $OUT_DIR


for data_id in "${DATA_LIST[@]}"; do
  for seed in {1:50}; do
    # Defind job name for this chromosome
    JOBN="D"$data_id"_S"$seed
    OUTF=$LOGS"/"$JOBN".out"
    ERRF=$LOGS"/"$JOBN".err"
    # Assemble slurm order for this job
    ORD=$ORDP -J $JOBN -o $OUTF -e $ERRF --export=data_id=$data_id,seed=$seed /?/submit.sh
  done
done
    