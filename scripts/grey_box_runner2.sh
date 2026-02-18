#!/bin/bash -l
#SBATCH --job-name="grey_box_jedi_test"
#SBATCH --output=scripts/logs/sched_jedi_%A_%a.log
#SBATCH --partition=nvl
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6

#SBATCH --array=0-7%2

#SBATCH --mail-type=end
#SBATCH --mail-user=mohata1@jh.edu

# activate python env
source $HOME/scratchcpriebe1/MO/network_manifolds/treehouse/bin/activate 

echo "Doing inference for Jedi model $SLURM_ARRAY_TASK_ID"

# run script
# python $HOME/scratchcpriebe1/MO/network_manifolds/scripts/greyBox_experiments.py --id "grey_wizard_$SLURM_ARRAY_TASK_ID" --do_inference --n_query 1 --n_replicate 10 --model_name greyUnitTest --context_file scripts/context/wizard.txt --context_line $SLURM_ARRAY_TASK_ID > scripts/logs/grey_wizard_m01_$SLURM_ARRAY_TASK_ID.out 2>&1

nohup python ~/scratchcpriebe1/MO/network_manifolds/scripts/greyBox_experiments.py --id jedi_$SLURM_ARRAY_TASK_ID --concat_method question --do_inference --pipe_inference --embed_output --n_query 100 --n_replicate 50 --max_length 256 --model_name grey_box_0 --context_file ~/scratchcpriebe1/MO/network_manifolds/scripts/context/jedi.txt --context_line $SLURM_ARRAY_TASK_ID > scripts/grey_jedi_01_$SLURM_ARRAY_TASK_ID.out 2>&1

wait
echo "Jedi model $SLURM_ARRAY_TASK_ID done"