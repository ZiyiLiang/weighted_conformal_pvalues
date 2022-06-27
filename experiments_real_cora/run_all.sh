#!/bin/bash

DATA_LIST=(0 1 2 3 4 5 6)

#DATA_LIST=(5)

# Slurm parameters
MEMO=5G                             # Memory required (5GB)
TIME=12:00:00                      # Time required (12 hours)
CORE=1                               # Cores required (1)

# Assemble order prefix
# ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

LOGS="/home1/ziyilian/CLRA/realdata/logs"
mkdir -p $LOGS

OUT_DIR="/home1/ziyilian/CLRA/realdata/results/"
mkdir -p $OUT_DIR


for data_id in "${DATA_LIST[@]}"; do
  for seed in {1..20}; do
    # Defind job name for this chromosome
    JOBN="D"$data_id"_S"$seed
    OUTF=$LOGS"/"$JOBN".out"
    ERRF=$LOGS"/"$JOBN".err"
    # Assemble slurm order for this job
    sbatch --mem=$MEMO --nodes=1 --ntasks=1 --cpus-per-task=1 --time=$TIME -J $JOBN -o $OUTF -e $ERRF --export=data_id=$data_id,seed=$seed /home1/ziyilian/CLRA/realdata/code/submit.sh
  done
done
    
