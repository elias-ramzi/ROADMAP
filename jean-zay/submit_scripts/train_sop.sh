#!/bin/bash
#SBATCH --job-name=train_sop     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=train_sop%j.out # output file name
#SBATCH --error=train_sop%j.ouy  # error file name

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.7.1

python marginap/run.py \
'dataset.SOPDataset.data_dir=${env:$SCRATCH}' \
'experience.experiment_name=sop_marginap_${general.sampler.HierarchicalSampler.batch_size}_t${loss.losses.MarginAP.tau}_m${loss.losses.MarginAP.mu}' \
experience=jean_zay \
experience.seed=333 \
loss=marginap \
loss.losses.MarginAP.tau=1 \
loss.losses.MarginAP.mu=0.025 \
general.max_iter=100 \
'optimizer.scheduler.MultiStepLR.milestones=[50]'
