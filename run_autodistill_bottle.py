# Import necessary libraries with error handling
import os
import re
import torch
import cv2
from utils.check_labels import *

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from autodistill_florence_2 import Florence2
# from autodistill_grounded_sam_2 import GroundedSAM2
import matplotlib.pyplot as plt
from utils.config_wood import *
import wandb

tag = "Florence"
#tag = "DINO"
# Initialize wandb
wandb.login()
wandb.init(project="auto_label", name=f"Wood {tag}", tags=f"{tag}")  # Updated project and run name
reset_folders(DATASET_DIR_PATH, "results")
sahi=False
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

plt.ion()
sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
plt.savefig("results/sample_images_grid.png")

# Define ontology
ont_list = ont_list = {
    "diffrent color":"color",
    "multiple defects appearing together on the surface":"combined",
    "visible hole, puncture, or gap in the material":"hole",
    "liquid or water":"liquid",
    "scratch abresion line crack":"scratch"
}
table = wandb.Table(columns=["prompt", "caption"])
for key, value in ont_list.items():
    table.add_data(key, value)

# Log the table
wandb.log({"Prompt Table": table})

print(ont_list)

ontology = CaptionOntology(ont_list)
# Initiate base model and autolabel
if tag == "DINO":
    base_model = GroundingDINO(ontology=CaptionOntology(ont_list))
if tag == "Florence":
    base_model = Florence2(ontology=CaptionOntology(ont_list))
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".jpg",
    output_folder=DATASET_DIR_PATH
)
# Log model settings
wandb.config.update({
    "model": (f"{base_model.__class__.__name__}"),
    "ontology": ont_list,
    "input_folder": IMAGE_DIR_PATH,
    "output_folder": DATASET_DIR_PATH,
    "sahi": sahi
})

dataset2 = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH
)
print("Dataset size:", len(dataset))

# Call the function to plot annotated images
plot_annotated_images(dataset, SAMPLE_SIZE, "results/sample_annotated_images_grid.png")

# evaluate the dataset
#update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
print(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
print(gt_dataset)
#summarize
summarize_annotation_distributions(gt_dataset)
compare_classes(gt_dataset, dataset)
compare_image_keys(gt_dataset, dataset)
evaluate_detections(dataset, gt_dataset)
compare_plot(dataset,gt_dataset)

wandb.finish()