#!/bin/bash
#SBATCH --job-name=SOTA     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=10:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0-1

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


name[0]='experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_2048'
transform[0]=sop
embed_dim[0]=2048
# norm_features[0]=False
# pooling[0]=avg
# memory[0]=null


name[1]='experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_patch256'
transform[1]=sop_big
embed_dim[1]=512
# norm_features[1]=False
# pooling[1]=avg
# memory[1]=null



sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
${name[${SLURM_ARRAY_TASK_ID}]} \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop_sota_ablation_study' \
experience.max_iter=100 \
optimizer=sop_small \
experience.num_workers=10 \
dataset.sampler.kwargs.batch_size=128 \
dataset.sampler.kwargs.batches_per_super_pair=10 \
loss=marginap_sigmo_abs \
transform=${transform[${SLURM_ARRAY_TASK_ID}]} \
model.kwargs.embed_dim=${embed_dim[${SLURM_ARRAY_TASK_ID}]}
