#!/bin/bash -l
#SBATCH --job-name="grey_box_wizard_test"
#SBATCH --output=scripts/logs/sched_wizard_%A_%a.log
#SBATCH --partition=nvl
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6

#SBATCH --mail-type=END
#SBATCH --mail-user=mohata1@jh.edu

# activate python env
source $HOME/scratchcpriebe1/MO/network_manifolds/treehouse/bin/activate 

echo "Getting MDS for q = 5, 10, 20, 50, 100"

nohup python ~/scratchcpriebe1/MO/network_manifolds/scripts/greyBox_mds_quick.py > scripts/mds_log.out 2>&1

wait
echo "Wizard model $SLURM_ARRAY_TASK_ID done"