#!/bin/bash
#SBATCH --job-name=INLA     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=30:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=INLA%A_%a.out # output file name
#SBATCH --error=INLA%A_%a.err  # error file name
#SBATCH --array=0-1

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.7.1

export TORCH_HOME=${SCRATCH}/pretrained_models

model[0]=deit
model[1]=resnet

opt[0]=inat_adamw
opt[1]=inaturalist

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_${model.kwargs.backbone_name}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat' \
experience.max_iter=100 \
optimizer=${opt[${SLURM_ARRAY_TASK_ID}]} \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=128 \
transform=inaturalist \
model=${model[${SLURM_ARRAY_TASK_ID}]} \
loss=smoothap_abs
