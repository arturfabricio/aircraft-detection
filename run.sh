#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1" 
#BSUB -J airJob
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."
python3 src/main.py
echo "Finished running scripts...."










