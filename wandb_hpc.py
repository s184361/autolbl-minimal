import wandb
import os
import re
import torch
import cv2
from utils.check_labels import *
from utils.config_hpc import *

tag = "Florence"
# Initialize wandb
wandb.login()
wandb.init(project="auto_label", name=f"Single {tag}", tags=f"{tag}") 

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH
)

print("Dataset size:", len(dataset))

# Log the size of the dataset
wandb.log({"dataset_size": len(dataset)})
# Save results images to wandb
plot_annotated_images(dataset, SAMPLE_SIZE, f"{RESULTS_DIR_PATH}/sample_annotated_images_grid.png")

# Evaluate the dataset
#update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
compare_classes(gt_dataset, dataset)
#compare_image_keys(gt_dataset, dataset)
evaluate_detections(dataset, gt_dataset,RESULTS_DIR_PATH)


#compare_plot(dataset, gt_dataset,results_dir=RESULTS_DIR_PATH)


# Finish the wandb run
wandb.finish()