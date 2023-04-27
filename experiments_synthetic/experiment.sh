#!/bin/bash

#module load gcc/4.9.4
#module load gcc/8.3.0
module load gcc/9.2.0
#module load intel/18.0.4
module load intel/19.0.4
 
module load python/3.7.6

python3 experiment.py $1 $2 $3 $4 $5 $6 $7

