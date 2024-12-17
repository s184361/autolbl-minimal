import os

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
DATASET_DIR_PATH = f"{HOME}/dataset"

# Plot sample images
SAMPLE_SIZE = 6
SAMPLE_GRID_SIZE = (3, 2)
SAMPLE_PLOT_SIZE = (16, 16)

PROMPT = "wood defect: knot, crack, resin, marrow, etc"
