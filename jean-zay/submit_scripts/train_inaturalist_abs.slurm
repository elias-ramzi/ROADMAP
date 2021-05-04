#!/bin/bash
#SBATCH --job-name=INLA     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=05:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=INLA%A_%a.out # output file name
#SBATCH --error=INLA%A_%a.err  # error file name
#SBATCH --array=0-5

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


alpha[0]=1.0
alpha[1]=0.9
alpha[2]=0.8
alpha[3]=1.0
alpha[4]=0.9
alpha[5]=0.8
alpha[6]=1.0
alpha[7]=0.9
alpha[8]=0.8

beta[0]=0.5
beta[1]=0.5
beta[2]=0.5
beta[3]=0.6
beta[4]=0.6
beta[5]=0.6
beta[6]=0.7
beta[7]=0.7
beta[8]=0.7


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}_a${loss.1.kwargs.pos_margin}_b${loss.1.kwargs.neg_margin}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_smoothabs_grid' \
experience.max_iter=20 \
optimizer=inaturalist \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=128 \
transform=inaturalist \
loss=smoothap_abs \
loss.1.kwargs.pos_margin=${alpha[${SLURM_ARRAY_TASK_ID}]} \
loss.1.kwargs.neg_margin=${beta[${SLURM_ARRAY_TASK_ID}]}
