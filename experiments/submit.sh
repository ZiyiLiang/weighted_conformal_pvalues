#!/bin/bash

#source C:/Users/liang/anaconda3/etc/profile.d/conda.sh
source /spack/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate py37

#python C:/Users/liang/OneDrive/Desktop/CLRA/codes/realdata/code/experiment.py $data_id $seed
#python -W ignore $HOME/CLRA/realdata/code/experiment.py $1 $2
python -W ignore $HOME/CLRA/realdata/code/experiment.py $data_id $seed
