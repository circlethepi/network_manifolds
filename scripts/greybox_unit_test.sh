#!/bin/bash -l
#SBATCH --job-name="grey_box_pipe_test"
#SBATCH --output=scripts/logs/sched_pipe.log
#SBATCH --partition=nvl
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohata1@jh.edu

source $HOME/scratchcpriebe1/MO/network_manifolds/treehouse/bin/activate 

nohup python ~/scratchcpriebe1/MO/network_manifolds/scripts/greyBox_experiments.py --id greyBox_pipe --do_inference --pipe_inference --embed_output --n_query 1 --n_replicate 10 --model_name greyUnitTest > scripts/logs/greyTest_pipe.out 2>&1