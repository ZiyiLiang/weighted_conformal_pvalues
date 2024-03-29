#!/bin/bash

# Parameters
SETUP="1"

if [[ $SETUP == 1 ]]; then
  DATA_LIST=("circles-mixed")
  N_LIST=(1000)
  P_LIST=(1000)
  A_LIST=(0.7)
  PURITY_LIST=(0.5 0.75 0.9)
  NUM_MODELS_LIST=(1 2 3 4 5 10 20 50 100)
  SEED_LIST=$(seq 1 100)

fi

# Slurm parameters
MEMO=5G                             # Memory required (5 GB)
TIME=00-00:30:00                    # Time required (30 m)
CORE=1                              # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME
#ORDP="sbatch --account=sesia_658 --partition=sesia,shared --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs"
mkdir -p $LOGS
mkdir -p $LOGS"/setup_greedy"$SETUP

OUT_DIR="results"
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR"/setup_greedy"$SETUP

# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  for DATA in "${DATA_LIST[@]}"; do

    for N in "${N_LIST[@]}"; do
      for P in "${P_LIST[@]}"; do
        for A in "${A_LIST[@]}"; do
          for PURITY in "${PURITY_LIST[@]}"; do
            for NUM_MODELS in "${NUM_MODELS_LIST[@]}"; do
              
              JOBN="setup_greedy"$SETUP"/"$DATA"_n"$N"_p"$P"_a"$A"_purity"$PURITY"_numodels"$NUM_MODELS"_seed"$SEED
              OUT_FILE=$OUT_DIR"/"$JOBN".txt"
              COMPLETE=0
              #ls $OUT_FILE
              if [[ -f $OUT_FILE ]]; then
                COMPLETE=1
              fi

              if [[ $COMPLETE -eq 0 ]]; then
                # Script to be run
                SCRIPT="experiment_greedy.sh $SETUP $DATA $N $P $A $PURITY $NUM_MODELS $SEED"
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
      done
    done
  done
done
