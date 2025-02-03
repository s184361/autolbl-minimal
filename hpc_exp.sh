#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J HPC_defects
### -- ask for number of cores (default: 1) -- 
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=16GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 16.5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s184361@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
# module load cuda
module load cuda
source /work3/s184361/miniconda3/bin/activate
conda activate alenv

cd /zhome/4a/b/137804/Desktop/autolbl
conda activate alenv

# Function to clear GPU memory
clear_gpu() {
    python -c "import torch; torch.cuda.empty_cache()"
    sleep 5
}

#python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section default --model DINO --tag default
#python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section bottle --model DINO --tag bottle
#python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section hpc --model DINO --tag hpc
#python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section wood --model DINO --tag wood
#python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --model Florence --tag Florence
#python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --model Combined --tag Combined
python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section defects --model DINO --tag defects
clear_gpu
#python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section defects --model Combined --tag defects
python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section defects --model Florence --tag defects
clear_gpu
python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section work3_defects --model DINO --tag defects
clear_gpu
#python run_any.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section work3_defects --model Florence --tag defects