#!/bin/bash
#SBATCH --job-name=train_inat     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=train_inat%j.out # output file name
#SBATCH --error=train_inat%j.err  # error file name
#SBATCH --array=0-7

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.7.1

export TORCH_HOME=${SCRATCH}/pretrained_models

loss[0]="smoothap"
loss[1]="smoothap"
loss[2]="adaptativeap"
loss[3]="adaptativeap"
loss[4]="marginap"
loss[5]="marginap"
loss[6]="contrastive"
loss[7]="contrastive"

batch_size[0]="32"
batch_size[1]="128"
batch_size[2]="32"
batch_size[3]="128"
batch_size[4]="32"
batch_size[5]="128"
batch_size[6]="32"
batch_size[7]="128"

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat' \
experience.max_iter=100 \
optimizer=inaturalist \
'optimizer.0.scheduler_on_epoch=null' \
'optimizer.1.scheduler_on_epoch=null' \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
transform=inaturalist \
loss=${loss[${SLURM_ARRAY_TASK_ID}]}
