#!/bin/bash

# Parameters
SETUP="1"

if [[ $SETUP == 1 ]]; then
  DATA_LIST=("circles-mixed")
  N_LIST=(200)
  P_LIST=(1000)
  A_LIST=(0.7)
  PURITY_LIST=(0.5 0.75 0.9)
  GAMMA_LIST=(1 0.1 0.01 0.005 0.002 0.001 0.0001 0.00001 0.000001)
  SEED_LIST=$(seq 1 100)

fi

# Slurm parameters
MEMO=5G                             # Memory required (5 GB)
TIME=00-00:20:00                    # Time required (20 m)
CORE=1                              # Cores required (1)

# Assemble order prefix
#ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME
ORDP="sbatch --account=sesia_658 --partition=sesia,shared --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs"
mkdir -p $LOGS
mkdir -p $LOGS"/setup_power"$SETUP

OUT_DIR="results"
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR"/setup_power"$SETUP

# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  for DATA in "${DATA_LIST[@]}"; do

    for N in "${N_LIST[@]}"; do
      for P in "${P_LIST[@]}"; do
        for A in "${A_LIST[@]}"; do
          for PURITY in "${PURITY_LIST[@]}"; do
            for GAMMA in "${GAMMA_LIST[@]}"; do
              
              JOBN="setup_power"$SETUP"/"$DATA"_n"$N"_p"$P"_a"$A"_purity"$PURITY"_gamma"$GAMMA"_seed"$SEED
              OUT_FILE=$OUT_DIR"/"$JOBN".txt"
              COMPLETE=0
              #ls $OUT_FILE
              if [[ -f $OUT_FILE ]]; then
                COMPLETE=1
              fi

              if [[ $COMPLETE -eq 0 ]]; then
                # Script to be run
                SCRIPT="experiment_power.sh $SETUP $DATA $N $P $A $PURITY $GAMMA $SEED"
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
    done
  done
done
