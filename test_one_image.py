#%%
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
from autodistill.utils import plot
from autodistill.core.composed_detection_model import ComposedDetectionModel
from autodistill_clip import CLIP

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

#base_model = GroundedSAM(ontology=ontology)
base_model = ComposedDetectionModel(
    detection_model=GroundedSAM(
        CaptionOntology({"defect": "defect"})
    ),
    classification_model=CLIP(
        CaptionOntology(ont_list)
    )
)
# Perform inference on the specified image
image_path = f"{HOME}/images/100000003.jpg"
result = base_model.predict(image_path)

# Plot the results
fig= plot(
    image=cv2.imread(image_path),
    classes=base_model.ontology.classes(),
    detections=result,
    raw=True
)
cv2.imwrite("results/annotated_image1.jpg", fig)
#%%
#extend raw image to see the annotations
image = cv2.imread(image_path)
#add white space around the image
image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255, 255, 255))
for detection in result:
    bbox = detection[0]
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    cv2.rectangle(image, (int(x) + 100, int(y) + 100), (int(x + w) + 100, int(y + h) + 100), (0, 255, 0), 2)
    cv2.putText(image, names[int(detection[3])], (int(x)+100, int(y)-10+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite("results/annotated_image.jpg", image)

#%%