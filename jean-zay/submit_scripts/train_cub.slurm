#!/bin/bash
#SBATCH --job-name=cub     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=10:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=cub%A_%a.out # output file name
#SBATCH --error=cub%A_%a.err  # error file name
#SBATCH --array=0-23

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

loss[0]="smoothap"
loss[1]="smoothap"
loss[2]="smoothap"
loss[3]="smoothap"
loss[4]="adaptativeap"
loss[5]="adaptativeap"
loss[6]="adaptativeap"
loss[7]="adaptativeap"
loss[8]="marginap"
loss[9]="marginap"
loss[10]="marginap"
loss[11]="marginap"
loss[12]="smoothap_abs"
loss[13]="smoothap_abs"
loss[14]="smoothap_abs"
loss[15]="smoothap_abs"
loss[16]="adaptativeap_abs"
loss[17]="adaptativeap_abs"
loss[18]="adaptativeap_abs"
loss[19]="adaptativeap_abs"
loss[20]="marginap_abs"
loss[21]="marginap_abs"
loss[22]="marginap_abs"
loss[23]="marginap_abs"

batch_size[0]="32"
batch_size[1]="64"
batch_size[2]="128"
batch_size[3]="224"
batch_size[4]="32"
batch_size[5]="64"
batch_size[6]="128"
batch_size[7]="224"
batch_size[8]="32"
batch_size[9]="64"
batch_size[10]="128"
batch_size[11]="224"
batch_size[12]="32"
batch_size[13]="64"
batch_size[14]="128"
batch_size[15]="224"
batch_size[16]="32"
batch_size[17]="64"
batch_size[18]="128"
batch_size[19]="224"
batch_size[20]="32"
batch_size[21]="64"
batch_size[22]="128"
batch_size[23]="224"

srun python margin_ap/single_experiment_runner.py \
--multirun \
'experience.experiment_name=cub_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_${model.kwargs.backbone_name}' \
experience.seed=250 \
'experience.log_dir=${env:HOME}/experiments/cub_final' \
experience.max_iter=201 \
experience.test_eval_freq=10 \
optimizer=cub \
optimizer.0.kwargs.lr=0.000001 \
transform=cub \
dataset=cub \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
loss=${loss[${SLURM_ARRAY_TASK_ID}]}
