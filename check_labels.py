# %%
import supervision as sv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

# %%
HOME = os.getcwd()
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"
GT_SEMANTIC_MAP_PATH = f"{HOME}/data/SemanticMaps"
SMS_PATH = f"{HOME}/data/Semantic Map Specification.txt"
# %%
image_path = f"{HOME}/dataset/images/100000000.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Load dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH,
)
# %%


with open("data/Semantic Map Specification.txt", "r") as file:
    content = file.read()
names = re.findall(r"name=([^\n]+)", content)
# %%
# load BB from data/Bounding Boxes/100000000_anno.txt
def read_labels_and_bboxes(file_path: str):
    labels_and_bboxes = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                label = parts[0]
                bbox = list(map(float, parts[1:]))
                labels_and_bboxes.append([label] + bbox)
    return np.array(labels_and_bboxes)


# Example usage
file_path = "data/Bounding Boxes/100000000_anno.txt"
labels_and_bboxes = read_labels_and_bboxes(file_path)
print(labels_and_bboxes)
#%%
# Get detections
detections = dataset.annotations[
    "/zhome/4a/b/137804/Desktop/autolbl/dataset/train/images/100000000.jpg"
]


# %%
# Load the BMP file

# Define the color to label mapping based on the provided labels
color_to_label = {
    (0, 255, 0): "Live_knot",
    (255, 0, 0): "Death_know",
    (255, 100, 0): "Knot_missing",
    (255, 175, 0): "knot_with_crack",
    (255, 0, 100): "Crack",
    (100, 0, 100): "Quartzity",
    (255, 0, 255): "resin",
    (0, 0, 255): "Marrow",
    (16, 255, 255): "Blue_stain",
    (0, 64, 0): "overgrown",
}
# %%

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Prepare YOLO annotations
height, width, _ = image.shape
yolo_annotations = []

for color, label in color_to_label.items():
    # Create a mask for the current color
    mask = cv2.inRange(image_rgb, np.array(color), np.array(color))
    # show the mask matplotlib
    plt.figure()
    plt.imshow(mask)

# %%
