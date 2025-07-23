#!/bin/bash -l
#SBATCH --job-name="sims-and-coordinates-all-layers"
#SBATCH --output=logs/sched_mds_layer_%A_%a.log
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6

#SBATCH --array=1-15%5

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohata1@jh.edu

# activate python env
source $HOME/scratchcpriebe1/MO/network_manifolds/treehouse/bin/activate
cd $HOME/scratchcpriebe1/MO/network_manifolds/scripts



echo "Running layer $SLURM_ARRAY_TASK_ID"

# run script (hopefully)
python ~/scratchcpriebe1/MO/network_manifolds/scripts/get_MDS_coordinates_mult_seeds.py --layer "$SLURM_ARRAY_TASK_ID" --act-count 1 --dir "scripts/sync/stab_multi0"


wait
echo "Layer $SLURM_ARRAY_TASK_ID done"
