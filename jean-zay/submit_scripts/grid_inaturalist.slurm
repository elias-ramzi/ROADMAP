#!/bin/bash
#SBATCH --job-name=grid_inat     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=05:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=grid_inat%A_%a.out # output file name
#SBATCH --error=grid_inat%A_%a.err  # error file name
#SBATCH --array=0-15

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

mu[0]=0.01
mu[1]=0.025
mu[2]=0.05
mu[3]=0.075
mu[4]=0.01
mu[5]=0.025
mu[6]=0.05
mu[7]=0.075
mu[8]=0.01
mu[9]=0.025
mu[10]=0.05
mu[11]=0.075
mu[12]=0.01
mu[13]=0.025
mu[14]=0.05
mu[15]=0.075

tau[0]=1.0
tau[1]=1.0
tau[2]=1.0
tau[3]=1.0
tau[4]=0.5
tau[5]=0.5
tau[6]=0.5
tau[7]=0.5
tau[8]=0.25
tau[9]=0.25
tau[10]=0.25
tau[11]=0.25
tau[12]=1.5
tau[13]=1.5
tau[14]=1.5
tau[15]=1.5

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_tau${loss.0.kwargs.tau}_mu${loss.0.kwargs.mu}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat' \
experience.max_iter=20 \
optimizer=inaturalist \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=224 \
transform=inaturalist \
model.kwargs.with_autocast=True \
loss=marginap \
loss.0.kwargs.tau=${tau[${SLURM_ARRAY_TASK_ID}]} \
loss.0.kwargs.mu=${mu[${SLURM_ARRAY_TASK_ID}]} 
