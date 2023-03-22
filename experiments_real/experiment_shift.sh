#!/bin/bash

module load python/3.7.6
ml gcc/8.3.0

python3 experiment_shift.py $1 $2 $3 $4

