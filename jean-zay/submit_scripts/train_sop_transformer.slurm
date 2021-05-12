#!/bin/bash
#SBATCH --job-name=TRSOP     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=TRSOP%A_%a.out # output file name
#SBATCH --error=TRSOP%A_%a.err  # error file name
#SBATCH --array=0-9

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

loss[0]="contrastive"
loss[1]="contrastive"
loss[2]="contrastive"
loss[3]="contrastive"
loss[4]="contrastive"
loss[5]="smoothap"
loss[6]="smoothap"
loss[7]="smoothap"
loss[8]="smoothap"
loss[9]="smoothap"


batch_size[0]="32"
batch_size[1]="64"
batch_size[2]="128"
batch_size[3]="224"
batch_size[4]="384"
batch_size[5]="32"
batch_size[6]="64"
batch_size[7]="128"
batch_size[8]="224"
batch_size[9]="384"

batches_per_super_pair[0]=40
batches_per_super_pair[1]=20
batches_per_super_pair[2]=10
batches_per_super_pair[3]=6
batches_per_super_pair[4]=4
batches_per_super_pair[5]=40
batches_per_super_pair[6]=20
batches_per_super_pair[7]=10
batches_per_super_pair[8]=6
batches_per_super_pair[9]=4


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_deit' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop_deit_baseline_middle' \
experience.max_iter=50 \
experience.num_workers=10 \
model=deit \
optimizer=sop_deit \
optimizer.0.kwargs.lr=0.00001 \
optimizer.0.name=AdamW \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
dataset.sampler.kwargs.batches_per_super_pair=${batches_per_super_pair[${SLURM_ARRAY_TASK_ID}]} \
loss=${loss[${SLURM_ARRAY_TASK_ID}]}
