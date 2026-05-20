#!/bin/bash -l

#SBATCH -job-name="finetune-yahoo9-test"
#SBATCH -output="~/scratchcpriebe1/MO/network_manifolds/jobs/nohup/finetune_yahoo_test.log"
#SBATCH -partition=a100
#SBATCH -t 00-06:00:00
#SBATCH -nodes=1
#SBATCH -gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohata1@jh.edu

# activate python env
source ~/scratchcpriebe1/MO/network_manifolds/treehouse/bin/activate 

# run script (hopefully)
nohup python ~/scratchcpriebe1/MO/network_manifolds/scripts/finetune_yahoo_test.py > jobs/nohup/job0.out 2>&1