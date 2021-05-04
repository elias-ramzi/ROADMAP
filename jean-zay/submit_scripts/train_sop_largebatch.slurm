#!/bin/bash
#SBATCH --job-name=train_sop     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:2                 # number of GPUs per node
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=20           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=train_sop%A_%a.out # output file name
#SBATCH --error=train_sop%A_%a.err  # error file name
#SBATCH --array=0-3

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

loss[0]="smoothap"
loss[1]="adaptativeap"
loss[2]="marginap"
loss[3]="contrastive"

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_schedule' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop' \
experience.max_iter=200 \
experience.resume=epoch_100.ckpt \
experience.force_lr=0.000001 \
'optimizer.0.scheduler_on_epoch=null' \
'optimizer.1.scheduler_on_epoch=null' \
dataset.sampler.kwargs.batch_size=384 \
loss=${loss[${SLURM_ARRAY_TASK_ID}]}
