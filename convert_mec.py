"""Utilities for converting between different annotation formats (masks, YOLO, Florence, COCO)."""

import os

import cv2
import numpy as np
import supervision as sv
from autodistill.helpers import split_data
import json


def convert_masks_to_yolo(input_dir, output_dir, classes):
    """
    Converts binary masks to YOLO segmentation format.

    Args:
        input_dir (str): Path to the root directory containing binary masks.
        output_dir (str): Path to save the YOLO annotations.
        classes (dict): Dictionary mapping subfolder names to class IDs (e.g., {'broken_large': 0, 'broken_small': 1}).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name, class_id in classes.items():
        mask_folder = os.path.join(input_dir, class_name)
        if not os.path.isdir(mask_folder):
            print(f"Skipping non-existent folder: {mask_folder}")
            continue

        for mask_file in os.listdir(mask_folder):
            if not mask_file.endswith((".png", ".jpg", ".jpeg")):
                continue

            mask_path = os.path.join(mask_folder, mask_file)
            class_name = os.path.basename(mask_folder)  # Get the class name from the folder

            # Remove "_mask" and add class name to the output name
            image_name = f"{os.path.splitext(mask_file)[0].replace('_mask', '')}_{class_name}.txt"
            output_path = os.path.join(output_dir, image_name)
            # Load binary mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                continue

            # Use supervision to convert mask to polygons
            polygons = sv.mask_to_polygons(mask)

            annotations = []
            height, width = mask.shape  # Get the dimensions of the mask
            for polygon in polygons:
                x_min, y_min, x_max, y_max = sv.polygon_to_xyxy(polygon)
                # Scale coordinates and dimensions to be between 0 and 1
                x_min /= width
                y_min /= height
                x_max /= width
                y_max /= height
                scaled_width = (x_max - x_min)
                scaled_height = (y_max - y_min)
                x_center = (x_min + scaled_width / 2)
                y_center = (y_min + scaled_height / 2)
                # Normalize polygon coordinates to be between 0 and 1
                polygon = np.array(polygon, dtype=np.float32)
                polygon[:, 0] /= width  # Normalize x coordinates
                polygon[:, 1] /= height  # Normalize y coordinates
                flattened_polygon = polygon.flatten().tolist()
                annotation = [class_id, x_center, y_center, scaled_width, scaled_height] + flattened_polygon
                annotations.append(annotation)

            # Write YOLO annotations to a text file
            with open(output_path, "w") as f:
                for annotation in annotations:
                    annotation_str = " ".join(map(str, annotation))
                    f.write(annotation_str + "\n")

            print(f"Saved annotation for {mask_file} to {output_path}")

    # Add empty annotations for images in test/good folder
    good_folder = os.path.join(input_dir, "test", "good")
    if os.path.isdir(good_folder):
        for good_file in os.listdir(good_folder):
            if not good_file.endswith((".png", ".jpg", ".jpeg")):
                continue

            good_image_name = f"{os.path.splitext(good_file)[0]}.txt"
            good_output_path = os.path.join(output_dir, good_image_name)

            # Write an empty annotation file
            with open(good_output_path, "w") as f:
                pass

            print(f"Saved empty annotation for {good_file} to {good_output_path}")


def save_test_images_with_parent_name(test_dir, images_output_dir):
    """
    Saves images from the test folder to a new directory, appending the parent folder name to the file name.

    Args:
        test_dir (str): Path to the test directory containing subfolders with images.
        images_output_dir (str): Path to save the renamed images.
    """
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)

    for parent_folder in os.listdir(test_dir):
        parent_path = os.path.join(test_dir, parent_folder)
        if not os.path.isdir(parent_path):
            continue

        for image_file in os.listdir(parent_path):
            if not image_file.endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(parent_path, image_file)
            image_name = os.path.splitext(image_file)[0]
            new_image_name = f"{image_name}_{parent_folder}.jpg"
            output_path = os.path.join(images_output_dir, new_image_name)

            # Copy the image to the new location with the updated name
            image = cv2.imread(image_path)
            if image is not None:
                cv2.imwrite(output_path, image)
                print(f"Saved image {image_file} as {new_image_name} to {output_path}")
            else:
                print(f"Failed to load image: {image_path}")


def annotations_for_good_pictures(input_dir, output_dir):
    """
    Creates empty annotation files for good pictures.

    Args:
        input_dir (str): Path to the directory containing good images.
        output_dir (str): Path to save the empty annotation files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    good_folder = os.path.join(input_dir, "good")
    if os.path.isdir(good_folder):
        for good_file in os.listdir(good_folder):
            if not good_file.endswith((".png", ".jpg", ".jpeg")):
                continue

            good_image_name = f"{os.path.splitext(good_file)[0]}_good.txt"
            good_output_path = os.path.join(output_dir, good_image_name)

            # Write an empty annotation file
            with open(good_output_path, "w") as f:
                pass

            print(f"Saved empty annotation for {good_file} to {good_output_path}")


def convert_yolo_to_florence(image_dir, annotation_dir, output_dir, classes=None, split=0.8):
    """
    Converts YOLO annotations to the Florence dataset format.

    Args:
        image_dir (str): Path to the directory containing images.
        annotation_dir (str): Path to the directory containing YOLO annotations.
        output_dir (str): Path to save the Florence dataset annotations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for annotation_file in os.listdir(annotation_dir):
        # Split the dataset into train, validation, and test sets
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith(".txt")]
        np.random.shuffle(annotation_files)
        train_split = int(len(annotation_files) * split)
        valid_split = int(len(annotation_files) * (split + (1 - split) / 2))

        train_files = annotation_files[:train_split]
        valid_files = annotation_files[train_split:valid_split]
        test_files = annotation_files[valid_split:]

        splits = {'train': train_files, 'valid': valid_files, 'test': test_files}

        for split_name, split_files in splits.items():
            split_output_dir = os.path.join(output_dir, split_name)
            if not os.path.exists(split_output_dir):
                os.makedirs(split_output_dir)

            for annotation_file in split_files:
                annotation_path = os.path.join(annotation_dir, annotation_file)
                image_name = os.path.splitext(annotation_file)[0] + ".jpg"
                image_path = os.path.join(image_dir, image_name)
                if not os.path.isfile(image_path):
                    print(f"Skipping missing image: {image_path}")
                    continue

                output_path = os.path.join(split_output_dir, annotation_file)
                with open(annotation_path, "r") as f:
                    lines = f.readlines()

                annotations = []
                for line in lines:
                    parts = line.strip().split(" ")
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    # Normalize between 0 and 999
                    x_min = int(x_min * 1000)
                    y_min = int(y_min * 1000)
                    annotations.append([class_id, x_min, y_min, x_max, y_max])

                with open(output_path, "w") as f:
                    for annotation in annotations:
                        class_id, x_min, y_min, x_max, y_max = annotation
                        if classes is not None:
                            class_name = list(classes.keys())[list(classes.values()).index(class_id)]
                        else:
                            class_name = str(class_id)
                        f.write(f"{class_name} <loc_{x_min}><loc_{y_min}><loc_{x_max}><loc_{y_max}>\n")

                print(f"Converted YOLO annotations for {annotation_file} to Florence format in {split_name} set")
        if not annotation_file.endswith(".txt"):
            continue

        annotation_path = os.path.join(annotation_dir, annotation_file)
        image_name = os.path.splitext(annotation_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        output_path = os.path.join(output_dir, annotation_file)
        with open(annotation_path, "r") as f:
            lines = f.readlines()

        annotations = []
        for line in lines:
            parts = line.strip().split(" ")
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            #normalize between 0 and 999
            x_min = int(x_min * 1000)
            y_min = int(y_min * 1000)
            annotations.append([class_id, x_min, y_min, x_max, y_max])

        with open(output_path, "w") as f:
            for annotation in annotations:
                class_id, x_min, y_min, x_max, y_max = annotation
                if classes is not None:
                    class_name = list(classes.keys())[list(classes.values()).index(class_id)]
                else:
                    class_name = str(class_id)
                f.write(f"{class_name} <loc_{x_min}><loc_{y_min}><loc_{x_max}><loc_{y_max}>\n")

        print(f"Converted YOLO annotations for {annotation_file} to Florence format")

if __name__ == "__main__":
    
    # Load config.json to get the data folder structure
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Get the local data folder path
    base_dir = config["local"]["data_folder"]
    
    # Split data into train/valid/test
    split_data(base_dir=base_dir, split_ratio=0.8)
    
    # Convert YOLO annotations to COCO format for train and valid sets
    dir_list = ["train", "valid"]
    for split_dir in dir_list:
        image_dir = os.path.join(base_dir, split_dir, "images")
        annotation_dir = os.path.join(base_dir, split_dir, "labels")
        yaml_dir = os.path.join(base_dir, "data.yaml")
        florence_output_dir = os.path.join(base_dir, "florence_annotations", split_dir)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(florence_output_dir):
            os.makedirs(florence_output_dir)
        
        # Load the dataset
        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=image_dir,
            annotations_directory_path=annotation_dir,
            data_yaml_path=yaml_dir
        )
        
        # Save as COCO format
        dataset.as_coco(
            images_directory_path=florence_output_dir,
            annotations_path=os.path.join(base_dir, "florence_annotations", f"{split_dir}_annotations.coco.json")
        )
        
        print(f"Converted {split_dir} set to COCO format")
