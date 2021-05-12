#!/bin/bash
#SBATCH --job-name=grid_tau     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=grid_tau%A_%a.out # output file name
#SBATCH --error=grid_tau%A_%a.err  # error file name
#SBATCH --array=0-7

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

tau[0]=0.5
tau[1]=0.75
tau[2]=1.0
tau[3]=1.25
tau[4]=1.5
tau[5]=1.75
tau[6]=2.0
tau[7]=0.25

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_tau${loss.0.kwargs.tau}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/simple_grid_tau' \
experience.max_iter=200 \
experience.num_workers=10 \
dataset.sampler.kwargs.batch_size=128 \
loss=adaptativeap \
loss.0.kwargs.tau=${tau[${SLURM_ARRAY_TASK_ID}]}
