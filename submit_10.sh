#!/bin/bash
# filepath: /zhome/4a/b/137804/Desktop/autolbl/submit_10.sh

# Script to submit opt_ax.sh 10 times to the LSF queue

echo "Starting submission of 10 jobs..."

for i in {1..10}
do
  echo "Submitting job $i of 10..."
  bsub < /zhome/4a/b/137804/Desktop/autolbl/opt_ax.sh
  
  # Add a short delay between submissions to avoid overwhelming the system
  sleep 2
done

echo "All 10 jobs have been submitted."