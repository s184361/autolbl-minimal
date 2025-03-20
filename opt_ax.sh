#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J opt_ax_wood_bert_pad_rand
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 3:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=8GB]"
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

# here follow the commands you want to execute with input.in as the input file

module load cuda
source /work3/s184361/miniconda3/bin/activate
conda activate alenv

# Create W&B directories in work3
mkdir -p /work3/s184361/wandb_data/wandb        # For logs
mkdir -p /work3/s184361/wandb_data/cache        # For artifacts cache
mkdir -p /work3/s184361/wandb_data/config       # For config files
mkdir -p /work3/s184361/wandb_data/staging      # For staging artifacts
mkdir -p /work3/s184361/wandb_data/artifacts    # For downloaded artifacts

# Set W&B environment variables
export WANDB_DIR=/work3/s184361/wandb_data/wandb
export WANDB_CACHE_DIR=/work3/s184361/wandb_data/cache
export WANDB_CONFIG_DIR=/work3/s184361/wandb_data/config
export WANDB_DATA_DIR=/work3/s184361/wandb_data/staging
export WANDB_ARTIFACT_DIR=/work3/s184361/wandb_data/artifacts

cd /zhome/4a/b/137804/Desktop/autolbl

# Function to clear GPU memory
clear_gpu() {
    python -c "import torch; torch.cuda.empty_cache()"
    sleep 5
    wandb artifact cache cleanup --remove-temp 2    
    wandb sync --clean --clean-force --clean-old-hours 2
}

# Monitor GPU usage with bnvtop for this job
echo "Running bnvtop $LSB_JOBID to monitor GPU usage"
bnvtop $LSB_JOBID &

clear_gpu
python test_opt_ax.py --n_trials=1 --randomize=False --ds_name=bottle --model=Florence --optimizer=ax --encoding_type=bert
python test_opt_ax.py --n_trials=1 --randomize=False --ds_name=bottle --model=DINO --optimizer=ax --encoding_type=bert

#bsub -v "MODEL=Florence DINO" < opt_ax.sh

