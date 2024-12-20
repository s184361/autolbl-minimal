import os

HOME = os.getcwd()

# Define paths and parameters
IMAGE_DIR_PATH = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/images"
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"
GT_ANNOTATIONS_DIRECTORY_PATH = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/yolo_annotations"
GT_IMAGES_DIRECTORY_PATH = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/images"
GT_DATA_YAML_PATH = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/data.yaml"
DATASET_DIR_PATH = f"{HOME}/dataset"
# Plot sample images
SAMPLE_SIZE = 6
SAMPLE_GRID_SIZE = (3, 2)
SAMPLE_PLOT_SIZE = (16, 16)

# PROMPT = "anomaly"
# PROMPT = "anomaly defect scrach speck miscoloration"
#PROMPT = "defects are irregularities that can affect the material's appearance, strength, or utility"
PROMPT = "defects are irregularities that can affect the appearance, strength, or utility"