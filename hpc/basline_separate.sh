#!/bin/bash
# filepath: /zhome/4a/b/137804/Desktop/autolbl/submit_jobs.sh

# Create tmp directory if it doesn't exist
mkdir -p /zhome/4a/b/137804/Desktop/autolbl/hpc/tmp

# Common bsub parameters for all jobs
COMMON_PARAMS="-q gpua100 -n 4 -gpu \"num=1:mode=exclusive_process\" -W 24:00 -R \"rusage[mem=4GB]\" -B -N -R \"span[hosts=1]\""

# Function to submit a job with a given command and name
submit_job() {
  local cmd="$1"
  local job_name="$2"
  
  echo "Submitting job: $job_name"
  
  # Create a script file in the hpc/tmp directory
  SCRIPT_FILE="/zhome/4a/b/137804/Desktop/autolbl/hpc/tmp/${job_name}_$(date +%Y%m%d%H%M%S).sh"
  
  # Write the job script
  cat > "$SCRIPT_FILE" << EOL
#!/bin/sh
### General options 
#BSUB -q gpua100
#BSUB -J ${job_name}
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -B
#BSUB -N
#BSUB -oo gpu_%J.out
#BSUB -eo gpu_%J.err
#BSUB -R "span[hosts=1]"

module load cuda
source /work3/s184361/miniconda3/bin/activate
conda activate alenv

cd /zhome/4a/b/137804/Desktop/autolbl
conda activate alenv

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Run the command
${cmd}
EOL

  # Make the script executable
  chmod +x "$SCRIPT_FILE"
  
  # Submit the job
  bsub < "$SCRIPT_FILE"
  
  # Wait a bit to avoid overloading the scheduler
  sleep 1
}

# Submit first batch of jobs (standard)
submit_job "python run_any3.py --section default --model Florence --tag default" "Florence_default"
submit_job "python run_any3.py --section bottle --model Florence --tag bottle --ontology \"broken_large: broken_large, broken_small: broken_small, contamination: contamination\"" "Florence_bottle"
submit_job "python run_any3.py --section wood --model Florence --tag wood --ontology \"color: color, combined: combined, hole: hole, liquid: liquid, scratch: scratch\"" "Florence_wood"
submit_job "python run_any3.py --model Florence --section wood --ontology \"anomaly defect: defect\" --tag defect" "Florence_defect"
submit_job "python run_any3.py --section tires --model Florence --tag tires --ontology \"car tires: car tires\"" "Florence_tires"

# Submit second batch of jobs (nms)
submit_job "python run_any3.py --section default --model Florence --tag default_nms --nms \"class_specific\"" "Florence_default_nms"
submit_job "python run_any3.py --section bottle --model Florence --tag bottle_nms --ontology \"broken_large: broken_large, broken_small: broken_small, contamination: contamination\"" "Florence_bottle_nms"
submit_job "python run_any3.py --section wood --model Florence --tag wood_nms --ontology \"color: color, combined: combined, hole: hole, liquid: liquid, scratch: scratch\"" "Florence_wood_nms"
submit_job "python run_any3.py --model Florence --section wood --ontology \"anomaly defect: defect\" --tag defect_nms" "Florence_defect_nms"
submit_job "python run_any3.py --section tires --model Florence --tag tires_nmss --ontology \"car tires: car tires\"" "Florence_tires_nms"

# Submit third batch of jobs (bag of words)
submit_job "python run_any3.py --section default --model Florence --tag default_bow --ontology \"BAG_OF_WORDS\"" "Florence_default_bow"
submit_job "python run_any3.py --section bottle --model Florence --tag bottle_bow --ontology \"BAG_OF_WORDS\"" "Florence_bottle_bow"
submit_job "python run_any3.py --section wood --model Florence --tag wood_bow --ontology \"BAG_OF_WORDS\"" "Florence_wood_bow"
submit_job "python run_any3.py --section tires --model Florence --tag tires_bow --ontology \"BAG_OF_WORDS\"" "Florence_tires_bow"

#submit_job "python run_any3.py --section work3 --model Florence --tag all_wood" "Florence_all_wood"
#submit_job "python run_any3.py --section work3 --model Florence --tag all_wood_nms" "Florence_all_wood_nms"
#submit_job "python run_any3.py --section work3 --model Florence --tag all_wood_bow --ontology \"BAG_OF_WORDS\"" "Florence_all_wood_bow"

#submit_job "python run_any3.py --section work3 --model MetaCLIP --tag all_wood" "MetaCLIP_all_wood"
#submit_job "python run_any3.py --section work3 --model MetaCLIP --tag all_wood_bow --ontology \"BAG_OF_WORDS\"" "MetaCLIP_all_wood_bow"
#submit_job "python run_any3.py --section wood --model MetaCLIP --tag all_wood" "MetaCLIP_wood"
#submit_job "python run_any3.py --section wood --model MetaCLIP --tag all_wood_bow --ontology \"BAG_OF_WORDS\"" "MetaCLIP_wood_bow"
#submit_job "python run_any3.py --section default --model MetaCLIP --tag all_wood" "MetaCLIP_default"
#submit_job "python run_any3.py --section default --model MetaCLIP --tag all_wood_bow --ontology \"BAG_OF_WORDS\"" "MetaCLIP_default_bow"