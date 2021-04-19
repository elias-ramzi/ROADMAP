#!/bin/bash
module purge
module load pytorch-gpu/py3/1.7.1 
module load timm
module load pytorch-metric-learning
module load faiss-gpu
module load hydra-core


export SOP_DATA_DIR=$SCRATCH/
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
export TMP_DIR=$JOBSCRATCH/
export TRAIN_DIR=experiments
