#!/bin/bash -l
#SBATCH --job-name="grey_box_jedi_test"
#SBATCH --output=scripts/logs/sched_jedi_%A_%a.log
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

echo "Doing inference for jedi model $SLURM_ARRAY_TASK_ID"

# run script
nohup python ~/scratchcpriebe1/MO/network_manifolds/scripts/greyBox_experiments.py --id jedi_$SLURM_ARRAY_TASK_ID --concat_method question --do_inference --embed_output --n_query 1000 --n_replicate 50 --max_length 256 --model_name grey_box_0 --context_file ~/scratchcpriebe1/MO/network_manifolds/scripts/context/jedi.txt --context_line $SLURM_ARRAY_TASK_ID > scripts/logs/grey_jedi_01_$SLURM_ARRAY_TASK_ID.out 2>&1

wait
echo "Jedi model $SLURM_ARRAY_TASK_ID done"