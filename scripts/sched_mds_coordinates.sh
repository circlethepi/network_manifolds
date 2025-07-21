#!/bin/bash -l
#SBATCH -job-name="sims-and-coordinates-all-layers"
#SBATCH -output="~/scratchcpriebe1/MO/network_manifolds/scripts/sched_mds_coordinates.log"
#SBATCH -partition=a100
#SBATCH -t 00-24:00:00
#SBATCH -nodes=1
#SBATCH -gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohata1@jh.edu

# activate python env
source ~/scratchcpriebe1/MO/network_manifolds/treehouse/bin/activate

for layer in {1..15}
do
    echo "Running layer $layer"
    
    # run script (hopefully)
    nohup python ~/scratchcpriebe1/MO/network_manifolds/scripts/get_MDS_coordinates_mult_seeds.py --layer "$layer" > "layer_$layer.out" 2 > &1 &
done

wait
echo "All jobs completed."
