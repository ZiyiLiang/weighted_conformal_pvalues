DATA=$1

mkdir -p results_hpc

rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup1/* results_hpc/setup1/
#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup2/* results_hpc/setup2/
#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup3/* results_hpc/setup3/
#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup4/* results_hpc/setup4/

#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_power1/* results_hpc/setup_power1/
#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_greedy1/* results_hpc/setup_greedy1/
#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_corr1/* results_hpc/setup_corr1/


#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_cv1/* results_hpc/setup_cv1/
#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_cv2/* results_hpc/setup_cv2/

#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_fdr1/* results_hpc/setup_fdr1/
#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_fdr2/* results_hpc/setup_fdr2/

#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_power_shift1/* results_hpc/setup_power_shift/

#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/weighted_conformal_pvalues/experiments_synthetic/results/setup_shift1/* results_hpc/setup_shift1/
