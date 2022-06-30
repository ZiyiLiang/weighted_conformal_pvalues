#!/bin/bash

SETUP=1

if [[ $SETUP == 1 ]]; then
#  DATA_LIST=("musk" "arrhythmia" "speech" "annthyroid" "mammography")
#  DATA_LIST=("images_flowers" "images_cars")
  DATA_LIST=("images_cars")
  #DATA_LIST=("mammography")
  #DATA_LIST=("shuttle")
  #DATA_LIST=("toxicity" "ad" "androgen" "rejafada")
  N_IN_LIST=(1000) # 1000 2000) # 5000)
  N_OUT_LIST=(2 5 10 20 30 50 75 100 150 200 500 1000) # 1000 2000) # 5000)
  SEED_LIST=$(seq 1 10)

fi

# Slurm parameters
MEMO=5G                             # Memory required (5 GB)
TIME=00-00:20:00                    # Time required (20 m)
CORE=1                              # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME
#ORDP="sbatch --account=sesia_658 --partition=sesia,shared --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs"
mkdir -p $LOGS

OUT_DIR="results"
mkdir -p $OUT_DIR

# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  for DATA in "${DATA_LIST[@]}"; do
    for N_IN in "${N_IN_LIST[@]}"; do
      for N_OUT in "${N_OUT_LIST[@]}"; do

        JOBN=$DATA"_nin"$N_IN"_nout"$N_OUT"_seed"$SEED
        OUT_FILE=$OUT_DIR"/"$JOBN".txt"
        COMPLETE=0
        #ls $OUT_FILE
        if [[ -f $OUT_FILE ]]; then
          COMPLETE=1
        fi

        if [[ $COMPLETE -eq 0 ]]; then
          # Script to be run
          SCRIPT="experiment.sh $DATA $N_IN $N_OUT $SEED"
          # Define job name for this chromosome
          OUTF=$LOGS"/"$JOBN".out"
          ERRF=$LOGS"/"$JOBN".err"
          # Assemble slurm order for this job
          ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
          # Print order
          echo $ORD
          # Submit order
          $ORD
          # Run command now
          #./$SCRIPT
        fi
      done
    done
  done
done
