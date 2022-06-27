#!/bin/bash

module load python/3.7.6
ml gcc/8.3.0

python3 experiment_fdr.py $1 $2 $3 $4 $5 $6 $7 $8 $9

