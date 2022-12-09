#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1" 
#BSUB -J airJob
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u s183685@student.dtu.dk
#BSUB -B 
#BSUB -N 
echo "Running script..."
python3 src/main.py
echo "Finished running scripts...."










