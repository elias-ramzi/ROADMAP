#!/bin/bash
#SBATCH --job-name=INLA     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=INLA%A_%a.out # output file name
#SBATCH --error=INLA%A_%a.err  # error file name
#SBATCH --array=0-3

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

batch_size[0]=32
batch_size[1]=64
batch_size[2]=128
batch_size[3]=224


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat' \
experience.max_iter=90 \
experience.num_workers=10 \
model.kwargs.with_autocast=True \
optimizer=inaturalist \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
transform=inaturalist \
loss=adaptativeap_abs \
loss.1.kwargs.pos_margin=0.9 \
loss.1.kwargs.neg_margin=0.6
