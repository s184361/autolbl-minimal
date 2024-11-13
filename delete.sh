#!/bin/bash

# Define the target directory
TARGET_DIR="/zhome/4a/b/137804/Desktop/autolbl/data/BoundingBoxes/"

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR not found"
    exit 1
fi

# Navigate to the data directory
cd "$TARGET_DIR" || { echo "Failed to navigate to $TARGET_DIR"; exit 1; }

# Iterate over all .bmp and .txt files
for file in *.bmp *.txt; do
    # Extract the numeric part of the filename
    num=$(echo "$file" | grep -oP '\d+')

    # Check if the numeric part is greater than or equal to 100400000
    if [[ "$num" -ge 100400000 ]]; then
        # Delete the file
        rm "$file"
    fi
done