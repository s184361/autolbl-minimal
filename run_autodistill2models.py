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
from autodistill.core.composed_detection_model import ComposedDetectionModel
from autodistill_clip import CLIP
#from IPython.display import Image

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
import matplotlib.pyplot as plt
plt.ion()
sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
plt.savefig("results/sample_images_grid.png")

# Define ontology
with open("data/Semantic Map Specification.txt", "r") as file:
    content = file.read()
names = re.findall(r"name=([^\n]+)", content)
names = [name.lower().replace("_", " ") for name in names]

#ont_list = {(f"{name} wood defect"): name for name in names}
ont_list = {(f"{name} defect"): name for name in names}
print(ont_list)
ontology = CaptionOntology(ont_list)

# Initiate base model and autolabel
DATASET_DIR_PATH = f"{HOME}/dataset"
#save copy of .bmp as .png

for root, dirs, files in os.walk(IMAGE_DIR_PATH):
    for file in files:
        if file.endswith('.bmp'):
            img = cv2.imread(os.path.join(root, file))
            cv2.imwrite(os.path.join(root, file.replace('.bmp', '.jpg')), img)
import clip
import torch
from PIL import Image
import os      

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

samclip = ComposedDetectionModel(
    detection_model=GroundingDINO(
        CaptionOntology({"defect": "defect"})
    ),
    classification_model=CLIP(
        CaptionOntology(ont_list)
    )
)
dataset = samclip.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".jpg",
    output_folder=DATASET_DIR_PATH
)

# Display dataset sample
#ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
#IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
#DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"

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
    #annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations, labels=labels)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations)
    images.append(annotated_image)


sv.plot_images_grid(images=images, titles=image_names, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
#save in high resolution
plt.savefig("results/sample_annotated_images_grid.png", dpi=1200)

#evaluate the dataset
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