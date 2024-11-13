# %%
import supervision as sv
import os
import numpy as np
import re
import yaml

# Print the version of the supervision package
print("Supervision version:", sv.__version__)

# %%
HOME = os.getcwd()
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"

# GT data
GT_ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/data/BoudingBoxes"
GT_IMAGES_DIRECTORY_PATH = f"{HOME}/images"
GT_DATA_YAML_PATH = f"{HOME}/data/data.yaml"
# %% Load SAM labels
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH,
)
# %% Update labels
# go through all txt files in labels directory

# Load the YAML file to get the mapping of labels to numbers
with open(GT_DATA_YAML_PATH, "r") as file:
    data_yaml = yaml.safe_load(file)
label_map = {label: idx for idx, label in enumerate(data_yaml["names"])}

# Directory containing the label files
labels_directory = os.path.join(GT_ANNOTATIONS_DIRECTORY_PATH)

# Iterate over all .txt files in the labels directory
for filename in os.listdir(labels_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(labels_directory, filename)

        # Read the file content
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Convert labels to numbers
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                label = parts[0].lower().replace("_", " ")
                bbox = parts[1:]
                label_number = label_map.get(label, -1)  # Use -1 if label not found
                new_line = f"{label_number} " + " ".join(bbox) + "\n"
                new_lines.append(new_line)

        # save as txt file without _anno in the title
        new_filename = re.sub(r"_anno", "", filename)
        new_file_path = os.path.join(labels_directory, new_filename)
        with open(new_file_path, "w") as file:
            file.writelines(new_lines)

# %% Load GT_WOOD dataset

gt_dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=GT_IMAGES_DIRECTORY_PATH,
    annotations_directory_path=GT_ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=GT_DATA_YAML_PATH,
)
gt_dataset
# %%
# compare gt classes with dataset classes
print("GT classes: ", gt_dataset.classes)
print("Dataset classes: ", dataset.classes)
print("GT classes not in dataset classes: ", gt_dataset.classes == dataset.classes)
# %%
# compare dataset image.keys with gt_dataset image.keys
print("Dataset image keys: ", dataset.images.keys())
print("GT image keys: ", gt_dataset.images.keys())
# print("GT image keys not in dataset image keys: ", gt_dataset.images.keys() == dataset.images.keys())

image_dataset = []
image_gt_dataset = []
for path, image, annotation in dataset:
    image_dataset.append(os.path.splitext(os.path.basename(path))[0])
for path, image, annotation in gt_dataset:
    image_gt_dataset.append(os.path.splitext(os.path.basename(path))[0])
#sort lists
image_dataset.sort()
image_gt_dataset.sort()
print("Dataset images: ", image_dataset)
print("GT images: ", image_gt_dataset)
print("GT images not in dataset images: ", image_gt_dataset == image_dataset)
# %%
# check if dataset.images.keys() corresponds to files in dataset.images_directory_path
for key in dataset.images.keys():
    if not os.path.exists(f"{IMAGES_DIRECTORY_PATH}/{key}"):
        print(f"File {key} not found in {IMAGES_DIRECTORY_PATH}")


# %% evaluate using from_detections

# assign 1 confidence to all annotations in dataset and gt_dataset
for key in dataset.annotations.keys():
    for i in range(len(dataset.annotations[key])):
        dataset.annotations[key].confidence= np.ones_like(dataset.annotations[key].class_id)
        #print(dataset.annotations[key].confidence)
for key in gt_dataset.annotations.keys():
    for i in range(len(gt_dataset.annotations[key])):
        gt_dataset.annotations[key].confidence= np.ones_like(gt_dataset.annotations[key].class_id)

# Check if all annotations in dataset have class_id
for key in dataset.annotations.keys():
    if not hasattr(dataset.annotations[key], 'class_id'):
        print(f"Annotation in dataset for {key} does not have class_id")

# Check if all annotations in gt_dataset have class_id
for key in gt_dataset.annotations.keys():
    if not hasattr(gt_dataset.annotations[key], 'class_id'):
        print(f"Annotation in gt_dataset for {key} does not have class_id")

# %%
predictions = []
predictions_keys = []
targets = []
targets_keys = []

for i, (image_path, image, annotation) in enumerate(dataset):
    key = os.path.basename(image_path)
    predictions.append(annotation)
    for j, (image_path_gt, image_gt, annotation_gt) in enumerate(gt_dataset):
        key_gt = os.path.basename(image_path_gt)
        key_gt = key_gt.replace(".png", ".jpg")
        
        if key == key_gt:
            print(key, key_gt)
            targets.append(annotation_gt)
            predictions_keys.append(key)
            targets_keys.append(key_gt)
            break
#%%
confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=dataset.classes,
    iou_threshold=0.5
)
#plot confusion matrix
confusion_matrix.plot(normalize=True)
print(confusion_matrix)
sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets
)
# %%