#!/bin/bash
#SBATCH --job-name=regresion_automobile     # create a short name for your job
#SBATCH --nodes=2               # node count
#SBATCH --ntasks-per-node=2     # total number of tasks per node
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

module load mamba-24

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT
export WORLD_SIZE=$SLURM_NPROCS
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
ifname=$(ip r list | grep 192.168.13.*/24 | awk '{print $3}')
export GLOO_SOCKET_IFNAME=$ifname

conda activate env_cluster

srun python Regresion_automobile.py
