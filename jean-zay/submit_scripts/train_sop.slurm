#!/bin/bash
#SBATCH --job-name=ERR     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=13:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%A_%a_err.out # output file name
#SBATCH --error=%A_%a_err.err  # error file name
#SBATCH --array=0-15

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


loss[0]=smoothap
seed[0]=0
with_autocast[0]=True

loss[1]=smoothap
seed[1]=1
with_autocast[1]=True

loss[2]=smoothap
seed[2]=2
with_autocast[2]=True

loss[3]=smoothap
seed[3]=3
with_autocast[3]=True


loss[4]=fastap
seed[4]=0
with_autocast[4]=True

loss[5]=fastap
seed[5]=1
with_autocast[5]=True

loss[6]=fastap
seed[6]=2
with_autocast[6]=True

loss[7]=fastap
seed[7]=3
with_autocast[7]=True


loss[8]=naverap
seed[8]=0
with_autocast[8]=True

loss[9]=naverap
seed[9]=1
with_autocast[9]=True

loss[10]=naverap
seed[10]=2
with_autocast[10]=True

loss[11]=naverap
seed[11]=3
with_autocast[11]=True


loss[12]=blackboxap
seed[12]=0
with_autocast[12]=False

loss[13]=blackboxap
seed[13]=1
with_autocast[13]=False

loss[14]=blackboxap
seed[14]=2
with_autocast[14]=False

loss[15]=blackboxap
seed[15]=3
with_autocast[15]=False


sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_seed${experience.seed}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=${seed[${SLURM_ARRAY_TASK_ID}]} \
'experience.log_dir=${env:SCRATCH}/experiments/sop_ranking_compa' \
experience.max_iter=100 \
experience.num_workers=10 \
experience.save_model=25 \
model.kwargs.with_autocast=${with_autocast[${SLURM_ARRAY_TASK_ID}]} \
optimizer=sop_small \
dataset.sampler.kwargs.batch_size=128 \
dataset.sampler.kwargs.batches_per_super_pair=10 \
loss=${loss[${SLURM_ARRAY_TASK_ID}]}
