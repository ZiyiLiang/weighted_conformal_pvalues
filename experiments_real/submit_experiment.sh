#!/bin/bash

if [[ $SETUP == 1 ]]; then
  DATA_LIST=("musk" "arrhythmia" "speech")
  N_LIST=(20 30 50 100 200 500 1000 2000 5000)
  SEED_LIST=$(seq 1 100)

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
mkdir -p $LOGS"/setup"$SETUP

OUT_DIR="results"
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR"/setup"$SETUP

# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  for DATA in "${DATA_LIST[@]}"; do
    for N in "${N_LIST[@]}"; do

      JOBN="setup"$SETUP"/"$DATA"_n"$N"_seed"$SEED
      OUT_FILE=$OUT_DIR"/"$JOBN".txt"
      COMPLETE=0
      #ls $OUT_FILE
      if [[ -f $OUT_FILE ]]; then
        COMPLETE=1
      fi

      if [[ $COMPLETE -eq 0 ]]; then
        # Script to be run
        SCRIPT="experiment.sh $DATA $N $SEED"
        # Define job name for this chromosome
        OUTF=$LOGS"/"$JOBN".out"
        ERRF=$LOGS"/"$JOBN".err"
        # Assemble slurm order for this job
        ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
        # Print order
        echo $ORD
        # Submit order
        #$ORD
        # Run command now
        #./$SCRIPT
      fi
    done
  done
done
