# Import necessary libraries with error handling
import os
import re
import torch
import cv2
from utils.check_labels import *

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
# from autodistill_grounded_sam_2 import GroundedSAM2

from utils.config2 import *
import wandb

wandb.login()
wandb.init()
# Delete dataset and results folders if they exist
#reset_folders(DATASET_DIR_PATH, "results")

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())
# Display image sample
image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["bmp", "jpg", "jpeg", "png"]
)
print('Image count:', len(image_paths))


titles = [image_path.stem for image_path in image_paths[:SAMPLE_SIZE]]
images = [cv2.imread(str(image_path)) for image_path in image_paths[:SAMPLE_SIZE]]
import matplotlib.pyplot as plt
plt.ion()
sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
plt.savefig("results/sample_images_grid.png")

# Define ontology
ont_list = {
    "big part broken off": "broken_large",
    "small part broken off": "broken_small",
    "contamination": "contamination",
}

print(ont_list)
ontology = CaptionOntology(ont_list)

base_model = GroundingDINO(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=DATASET_DIR_PATH
)

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH
)
print("Dataset size:", len(dataset))

# Call the function to plot annotated images
plot_annotated_images(dataset, SAMPLE_SIZE, "results/sample_annotated_images_grid.png")

# evaluate the dataset
#update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
compare_classes(gt_dataset, dataset)
compare_image_keys(gt_dataset, dataset)
evaluate_detections(dataset, gt_dataset)
compare_plot(dataset,gt_dataset)

