# Import necessary libraries with error handling
import os
import re
import torch
import cv2

try:
    import autodistill
    print("autodistill installed successfully.")
except ImportError:
    print("Failed to import autodistill.")

try:
    import autodistill_grounded_sam
    print("autodistill-grounded-sam installed successfully.")
except ImportError:
    print("Failed to import autodistill-grounded-sam.")

try:
    import autodistill_yolov8
    print("autodistill-yolov8 installed successfully.")
except ImportError:
    print("Failed to import autodistill-yolov8.")

try:
    import roboflow
    print("roboflow installed successfully.")
except ImportError:
    print("Failed to import roboflow.")

try:
    import supervision as sv
    print("supervision installed successfully.")
except ImportError:
    print("Failed to import supervision.")

from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_yolov8 import YOLOv8
from IPython.display import Image

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())

HOME = os.getcwd()
print("Current working directory:", HOME)

# Define paths and parameters
VIDEO_DIR_PATH = f"{HOME}/videos"
IMAGE_DIR_PATH = f"{HOME}/images"
FRAME_STRIDE = 10

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
plt.savefig("sample_images_grid.png")

# Define ontology
with open("data/Semantic Map Specification.txt", "r") as file:
    content = file.read()
names = re.findall(r"name=([^\n]+)", content)
names = [name.lower().replace("_", " ") for name in names]

ont_list = {(f"{name} wood defect"): name for name in names}
print(ont_list)
ontology = CaptionOntology(ont_list)

# Initiate base model and autolabel
DATASET_DIR_PATH = f"{HOME}/dataset"
#save copy of .bmp as .png
import os

for root, dirs, files in os.walk(IMAGE_DIR_PATH):
    for file in files:
        if file.endswith('.bmp'):
            img = cv2.imread(os.path.join(root, file))
            cv2.imwrite(os.path.join(root, file.replace('.bmp', '.png')), img)

base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=DATASET_DIR_PATH
)

# Display dataset sample
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"

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
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations, labels=labels)
    images.append(annotated_image)


sv.plot_images_grid(images=images, titles=image_names, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
#save in high resolution
plt.savefig("sample_annotated_images_grid.png", dpi=1200)
# Train target model - YOLOv8
target_model = YOLOv8("yolov8n.pt")
target_model.train(DATA_YAML_PATH, epochs=10)

# Evaluate target model
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)

# Run Inference on a video
#INPUT_VIDEO_PATH = TEST_VIDEO_PATHS[0]
#OUTPUT_VIDEO_PATH = f"{HOME}/output.mp4"
#TRAINED_MODEL_PATH = f"{HOME}/runs/detect/train/weights/best.pt"
