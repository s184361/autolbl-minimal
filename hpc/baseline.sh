#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J opt_ax_wood_bert_pad_rand
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=4GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s184361@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gpu_%J.out
#BSUB -eo gpu_%J.err
#BSUB -R "span[hosts=1]"
# -- end of LSF options --

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

python run_any3.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section default --model DINO --tag default --nms 
python run_any3.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section bottle --model DINO --tag bottle
python run_any3.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section hpc --model DINO --tag all_wood
python run_any3.py --config /zhome/4a/b/137804/Desktop/autolbl/config.json --section wood --model DINO --tag wood
