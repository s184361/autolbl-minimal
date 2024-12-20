import wandb
import os
import re
import torch
import cv2
from utils.check_labels import *
from autodistill.detection import CaptionOntology
from autodistill_florence_2 import Florence2
from utils.config import *

# Initialize wandb
wandb.login()
wandb.init(project="autolbl")  # Specify your project name
reset_folders(DATASET_DIR_PATH, "results")

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())

# Display image sample
image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["bmp", "jpg", "jpeg", "png"]
)
print('Image count:', len(image_paths))

# Define ontology
with open("data/Semantic Map Specification.txt", "r") as file:
    content = file.read()
names = re.findall(r"name=([^\n]+)", content)
names = [name.lower().replace("_", " ") for name in names]
ont_list = {(f"{name} surface defect"): name for name in names}

print(ont_list)

# Log the ontology list
wandb.log({"ont_list": str(ont_list)})

convert_bmp_to_jpg(IMAGE_DIR_PATH)

# Initiate base model and autolabel
base_model = Florence2(ontology=CaptionOntology(ont_list))

# Log model settings
wandb.config.update({
    "model": "Florence2",
    "ontology": ont_list,
    "input_folder": IMAGE_DIR_PATH,
    "output_folder": DATASET_DIR_PATH,
})

# Label the dataset
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=DATASET_DIR_PATH,
    sahi=True
)

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH
)
print("Dataset size:", len(dataset))

# Call the function to plot annotated images
plot_annotated_images(dataset, SAMPLE_SIZE, "results/sample_annotated_images_grid.png")

# Evaluate the dataset
update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
compare_classes(gt_dataset, dataset)
compare_image_keys(gt_dataset, dataset)
evaluate_detections(dataset, gt_dataset)
compare_plot(dataset, gt_dataset)

# Log the size of the dataset
wandb.log({"dataset_size": len(dataset)})

# Save results images to wandb
for image_name in dataset.images.keys():
    image = dataset.images[image_name]
    wandb.log({f"annotated_image_{image_name}": wandb.Image(image)})

# Save the results folder images
results_folder_path = "results/sample_annotated_images_grid.png"
wandb.log({"sample_images": wandb.Image(results_folder_path)})

# Finish the wandb run
wandb.finish()