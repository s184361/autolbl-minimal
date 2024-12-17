# Import necessary libraries with error handling
import os
import re
import torch
import cv2
from check_labels import *
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_grounding_dino import GroundingDINO
from autodistill_yolov8 import YOLOv8
import matplotlib.pyplot as plt
from autodistill.core import EmbeddingOntologyImage
from autodistill.core.composed_detection_model import ComposedDetectionModel
from autodistill_clip import CLIP

# from IPython.display import Image

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())

HOME = os.getcwd()
print("Current working directory:", HOME)

# Define paths and parameters
IMAGE_DIR_PATH = f"{HOME}/images"
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"
GT_ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/data/BoudingBoxes"
GT_IMAGES_DIRECTORY_PATH = f"{HOME}/images"
GT_DATA_YAML_PATH = f"{HOME}/data/data.yaml"

# Display image sample
image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["bmp", "jpg", "jpeg", "png"]
)
print('Image count:', len(image_paths))

# Plot sample images
SAMPLE_SIZE = 6
SAMPLE_GRID_SIZE = (3, 2)
SAMPLE_PLOT_SIZE = (16, 16)

titles = [image_path.stem for image_path in image_paths[:SAMPLE_SIZE]]
images = [cv2.imread(str(image_path)) for image_path in image_paths[:SAMPLE_SIZE]]
plt.ion()
sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
plt.savefig("results/sample_images_grid.png")

# Define ontology
with open("data/Semantic Map Specification.txt", "r") as file:
    content = file.read()
names = re.findall(r"name=([^\n]+)", content)
names = [name.lower().replace("_", " ") for name in names]
# Embeding dir
EMBEDING_DIR_PATH = os.path.abspath(os.path.join(HOME, "Image_Embeddings"))
convert_bmp_to_jpg(EMBEDING_DIR_PATH)

# Update paths to use os.path.join for proper path formatting
images_to_classes = {
    "live knot": "100000010.jpg",
    "dead knot": "100100010.jpg",
    "knot missing": "101800000.jpg",
    "knot with crack": "100000082.jpg",
    "crack": "100500053.jpg",
    "quartzity": "100000001.jpg",
    "resin": "101100021.jpg",
    "marrow": "101900001.jpg",
    "overgrown": "139100026.jpg",
    "blue stain": "144100014.jpg",
}

# Add a check to verify all image files exist
for class_name, image_path in images_to_classes.items():
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found for {class_name}: {image_path}")


ontology = EmbeddingOntologyImage(images_to_classes)
classification_model = GroundingDINO(ontology=ontology)
# Initiate base model and autolabel
DATASET_DIR_PATH = f"{HOME}/dataset"

convert_bmp_to_jpg(IMAGE_DIR_PATH)

# Create a combined model that uses both GroundingDINO for detection and CLIP for classification

# Use the combined model to label your dataset
dataset = classification_model.label(
    input_folder=IMAGE_DIR_PATH, extension=".png", output_folder=DATASET_DIR_PATH
)

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH
)
print("Dataset size:", len(dataset))

image_names = list(dataset.images.keys())[:SAMPLE_SIZE]

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

images = []
for image_name in image_names:
    image = dataset.images[image_name]
    annotations = dataset.annotations[image_name]
    labels = [dataset.classes[class_id] for class_id in annotations.class_id]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=annotations)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations)
    images.append(annotated_image)


sv.plot_images_grid(images=images, titles=image_names, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
# save in high resolution
plt.savefig("results/sample_annotated_images_grid.png", dpi=1200)

# evaluate the dataset
update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
compare_classes(gt_dataset, dataset)
compare_image_keys(gt_dataset, dataset)
evaluate_detections(dataset, gt_dataset)

# Train target model - YOLOv8
target_model = YOLOv8("yolov8n.pt")
"""
target_model.train(DATA_YAML_PATH, epochs=10)

# Evaluate target model
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)

"""
