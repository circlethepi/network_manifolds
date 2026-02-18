#!/bin/bash -l
#SBATCH --job-name="grey_box_wizard_test"
#SBATCH --output=scripts/logs/sched_wizard_%A_%a.log
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6

#SBATCH --array=0-19

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohata1@jh.edu

# activate python env
source $HOME/scratchcpriebe1/MO/network_manifolds/treehouse/bin/activate 

echo "Doing inference for wizard model $SLURM_ARRAY_TASK_ID"

# run script
# python $HOME/scratchcpriebe1/MO/network_manifolds/scripts/greyBox_experiments.py --id "grey_wizard_$SLURM_ARRAY_TASK_ID" --do_inference --n_query 1 --n_replicate 10 --model_name greyUnitTest --context_file scripts/context/wizard.txt --context_line $SLURM_ARRAY_TASK_ID > scripts/logs/grey_wizard_m01_$SLURM_ARRAY_TASK_ID.out 2>&1

nohup python ~/scratchcpriebe1/MO/network_manifolds/scripts/greyBox_experiments.py --id wizard_$SLURM_ARRAY_TASK_ID --concat_method question --do_inference --embed_output --n_query 100 --n_replicate 50 --max_length 256 --model_name grey_box_0 --context_file ~/scratchcpriebe1/MO/network_manifolds/scripts/context/wizard.txt --context_line $SLURM_ARRAY_TASK_ID > scripts/logs/grey_wizard_01_$SLURM_ARRAY_TASK_ID.out 2>&1

wait
echo "Wizard model $SLURM_ARRAY_TASK_ID done"