#!/bin/bash
#SBATCH --job-name=ABS     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=06:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0-12

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


weight0[0]=0.5
weight1[0]=0.5
pos_margin[0]=0.9
neg_margin[0]=0.6

weight0[1]=0.9
weight1[1]=0.1
pos_margin[1]=0.9
neg_margin[1]=0.6

weight0[2]=0.1
weight1[2]=0.9
pos_margin[2]=0.9
neg_margin[2]=0.6

weight0[3]=0.3
weight1[3]=0.7
pos_margin[3]=0.9
neg_margin[3]=0.6

weight0[4]=0.7
weight1[4]=0.3
pos_margin[4]=0.9
neg_margin[4]=0.6


weight0[5]=0.5
weight1[5]=0.5
pos_margin[5]=0.9
neg_margin[5]=0.7

weight0[6]=0.5
weight1[6]=0.5
pos_margin[6]=0.5
neg_margin[6]=0.5

weight0[7]=0.5
weight1[7]=0.5
pos_margin[7]=0.9
neg_margin[7]=0.5

weight0[8]=0.5
weight1[8]=0.5
pos_margin[8]=0.9
neg_margin[8]=0.3


weight0[9]=0.6
weight1[9]=0.4
pos_margin[9]=0.9
neg_margin[9]=0.6

weight0[10]=0.4
weight1[10]=0.6
pos_margin[10]=0.9
neg_margin[10]=0.6


weight0[11]=0.8
weight1[11]=0.2
pos_margin[11]=0.9
neg_margin[11]=0.6

weight0[12]=0.2
weight1[12]=0.8
pos_margin[12]=0.9
neg_margin[12]=0.6


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}_w${loss.1.weight}_p${loss.1.kwargs.pos_margin}_n${loss.1.kwargs.neg_margin}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop_marginap_abs_grid' \
experience.max_iter=100 \
experience.num_workers=10 \
dataset.sampler.kwargs.batch_size=128 \
dataset.sampler.kwargs.batches_per_super_pair=10 \
loss=marginap_sigmo_abs \
optimizer=sop_small \
optimizer.0.kwargs.lr=0.00002 \
optimizer.1.kwargs.lr=0.00004 \
loss.0.weight=${weight0[${SLURM_ARRAY_TASK_ID}]} \
loss.1.weight=${weight1[${SLURM_ARRAY_TASK_ID}]} \
loss.1.kwargs.pos_margin=${pos_margin[${SLURM_ARRAY_TASK_ID}]} \
loss.1.kwargs.neg_margin=${neg_margin[${SLURM_ARRAY_TASK_ID}]}
