#!/bin/bash
#SBATCH --job-name=dwnlddata     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3         # le calcul va etre long
#SBATCH --partition=archive
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $STORE

curl -C - "http://imagenet.stanford.edu/internal/car196/cars_train.tgz" --output cars_train.tgz
