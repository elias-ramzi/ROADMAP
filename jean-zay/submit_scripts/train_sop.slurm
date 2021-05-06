#!/bin/bash
#SBATCH --job-name=train_sop     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=10:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=train_sop%j.out # output file name
#SBATCH --error=train_sop%j.err  # error file name

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

# loss[0]="smoothap"
# loss[1]="smoothap"
# loss[2]="smoothap"
# loss[3]="smoothap"

batch_size[0]="32"
batch_size[1]="64"
batch_size[2]="128"
batch_size[3]="224"

batches_per_super_pair[0]=24
batches_per_super_pair[1]=12
batches_per_super_pair[2]=6
batches_per_super_pair[3]=3

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_schedule' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop_smoothap_baseline' \
experience.max_iter=50 \
experience.num_workers=10 \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
dataset.sampler.kwargs.batches_per_super_pair=${batches_per_super_pair[${SLURM_ARRAY_TASK_ID}]} \
loss=smoothap
