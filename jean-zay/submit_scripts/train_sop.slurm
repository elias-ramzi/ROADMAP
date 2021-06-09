#!/bin/bash
#SBATCH --job-name=32     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=7:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%A_%a_err.out # output file name
#SBATCH --error=%A_%a_err.err  # error file name
#SBATCH --array=0-4

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


pos_margin[0]=0.9
neg_margin[0]=0.6
lambda[0]=0.1

pos_margin[1]=0.9
neg_margin[1]=0.6
lambda[1]=0.5

pos_margin[2]=0.9
neg_margin[2]=0.7
lambda[2]=1.0

pos_margin[3]=0.9
neg_margin[3]=0.5
lambda[3]=1.0

pos_margin[4]=0.9
neg_margin[4]=0.8
lambda[4]=1.0


sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_lambda${loss.1.weight}_pos${loss.1.kwargs.pos_margin}_neg${loss.1.kwargs.neg_margin}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop_ablation_sota' \
experience.max_iter=100 \
experience.num_workers=10 \
experience.save_model=50 \
optimizer=sop_small \
dataset.sampler.kwargs.batch_size=32 \
dataset.sampler.kwargs.batches_per_super_pair=20 \
loss=marginap_sigmo_abs \
loss.1.weight=${lambda[${SLURM_ARRAY_TASK_ID}]} \
loss.1.kwargs.pos_margin=${pos_margin[${SLURM_ARRAY_TASK_ID}]} \
loss.1.kwargs.neg_margin=${neg_margin[${SLURM_ARRAY_TASK_ID}]}
