#!/bin/bash
#SBATCH --job-name=I384     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:2                 # number of GPUs per node
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


batch_size[0]=384
embed_dim[0]=2048

batch_size[1]=384
embed_dim[1]=512

# batch_size[0]=224
# embed_dim[0]=512
#
# batch_size[1]=128
# embed_dim[1]=512
#
# batch_size[2]=32
# embed_dim[2]=512


sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}_${model.kwargs.embed_dim}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_final' \
experience.max_iter=90 \
experience.num_workers=10 \
optimizer=inaturalist \
model.kwargs.embed_dim=${embed_dim[${SLURM_ARRAY_TASK_ID}]} \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
transform=inaturalist \
loss=marginap_sigmo_abs \
+loss.0.kwargs.start=0.0 \
loss.0.kwargs.offset=1.0 \
loss.0.kwargs.tau=100 \
experience.resume=rolling.ckpt
