#!/bin/bash
#SBATCH --job-name=train_sop     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=train_sop%j.out # output file name
#SBATCH --error=train_sop%j.err  # error file name
#SBATCH --array=0-11

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.7.1

export TORCH_HOME=${SCRATCH}/pretrained_models

loss[0]="smoothap"
loss[1]="smoothap"
loss[2]="smoothap"
loss[3]="adaptativeap"
loss[4]="adaptativeap"
loss[5]="adaptativeap"
loss[6]="marginap"
loss[7]="marginap"
loss[8]="marginap"
loss[9]="contrastive"
loss[10]="contrastive"
loss[11]="contrastive"

batch_size[0]="32"
batch_size[1]="64"
batch_size[2]="128"
batch_size[3]="32"
batch_size[4]="64"
batch_size[5]="128"
batch_size[6]="32"
batch_size[7]="64"
batch_size[8]="128"
batch_size[9]="32"
batch_size[10]="64"
batch_size[11]="128"

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_schedule' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop' \
experience.max_iter=300 \
experience.resume=epoch_100.ckpt \
'optimizer.0.scheduler_on_epoch=null' \
'optimizer.1.scheduler_on_epoch=null' \
experience.force_lr=0.000001 \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
loss=${loss[${SLURM_ARRAY_TASK_ID}]}
