DATA=$1

mkdir -p results_hpc

rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_real/results/* results_hpc/