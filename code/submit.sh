#!/bin/bash

#source C:/Users/liang/anaconda3/etc/profile.d/conda.sh
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate py37

#python C:/Users/liang/OneDrive/Desktop/CLRA/codes/realdata/code/experiment.py $data_id $seed
echo "$1"
echo "$2"
python -W ignore /mnt/c/users/liang/OneDrive/Desktop/CLRA/codes/realdata/code/experiment.py $1 $2
