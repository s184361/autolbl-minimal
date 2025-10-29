"""Common utilities for dataset preparation and configuration management."""

import json
import os
import cv2
import numpy as np
from pathlib import Path
import supervision as sv


def get_base_dir(script_path):
    """
    Auto-detect base directory from script path.
    
    Args:
        script_path: Path(__file__) from the calling script
    
    Returns:
        Path object to base directory
    """
    script_path = Path(script_path).absolute()
    
    # If script is in utils folder, go up one level
    if script_path.parent.name == "utils":
        return script_path.parent.parent
    else:
        return script_path.parent


def update_config_section(section_name, section_data, config_path=None):
    """
    Add or update a section in config.json.
    
    Args:
        section_name: Name of the config section (e.g., 'local_wood')
        section_data: Dictionary with configuration data
        config_path: Optional path to config.json (auto-detected if None)
    """
    if config_path is None:
        # Find config.json in base directory
        base_dir = Path.cwd()
        config_path = base_dir / "config.json"
        
        if not config_path.exists():
            print(f"Error: config.json not found at {config_path}")
            return False
    
    # Load existing config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Add or update section
    config[section_name] = section_data
    
    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Updated config.json with '{section_name}' section")
    return True


def create_config_section(base_dir, dataset_name, images_dir, annotations_dir, data_yaml_path, prompt=None):
    """
    Create a standard configuration section for a dataset.
    
    Args:
        base_dir: Base directory path
        dataset_name: Name for the config section (e.g., 'local_wood')
        images_dir: Path to images directory
        annotations_dir: Path to annotations directory
        data_yaml_path: Path to data.yaml file
        prompt: Optional prompt text
    
    Returns:
        Dictionary with configuration data
    """
    base_dir_str = str(base_dir).replace("\\", "/")
    images_dir_str = str(images_dir).replace("\\", "/")
    annotations_dir_str = str(annotations_dir).replace("\\", "/")
    data_yaml_str = str(data_yaml_path).replace("\\", "/")
    
    if prompt is None:
        prompt = "defects are irregularities that can affect the appearance, strength, or utility"
    
    return {
        "HOME": base_dir_str,
        "IMAGE_DIR_PATH": images_dir_str,
        "ANNOTATIONS_DIRECTORY_PATH": f"{base_dir_str}/dataset/train/labels",
        "IMAGES_DIRECTORY_PATH": f"{base_dir_str}/dataset/train/images",
        "DATA_YAML_PATH": f"{base_dir_str}/dataset/data.yaml",
        "GT_ANNOTATIONS_DIRECTORY_PATH": annotations_dir_str,
        "GT_IMAGES_DIRECTORY_PATH": images_dir_str,
        "GT_DATA_YAML_PATH": data_yaml_str,
        "DATASET_DIR_PATH": f"{base_dir_str}/dataset",
        "RESULTS_DIR_PATH": f"{base_dir_str}/results",
        "SAMPLE_SIZE": 6,
        "SAMPLE_GRID_SIZE": [3, 2],
        "SAMPLE_PLOT_SIZE": [16, 16],
        "PROMPT": prompt
    }


def convert_bbox_annotation(bbox_line, image_width=None, image_height=None):
    """
    Convert bounding box annotation to YOLO format.
    Handles multiple input formats.
    
    Args:
        bbox_line: Line from annotation file
        image_width: Width of the image (optional for normalized coords)
        image_height: Height of the image (optional for normalized coords)
    
    Returns:
        Tuple of (class_name, x_center, y_center, width, height) or None
    """
    parts = bbox_line.strip().split()
    if len(parts) < 5:
        return None
    
    class_name = parts[0]
    # Handle comma as decimal separator (convert to period)
    x_min = float(parts[1].replace(',', '.'))
    y_min = float(parts[2].replace(',', '.'))
    x_max = float(parts[3].replace(',', '.'))
    y_max = float(parts[4].replace(',', '.'))
    
    # Calculate YOLO format (normalized coordinates)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return class_name, x_center, y_center, width, height


def convert_image_format(input_path, output_path, target_format='jpg'):
    """
    Convert image from one format to another.
    
    Args:
        input_path: Path to input image
        output_path: Path to output image
        target_format: Target format ('jpg', 'png', etc.)
    
    Returns:
        True if successful, False otherwise
    """
    image = cv2.imread(str(input_path))
    if image is None:
        return False
    
    cv2.imwrite(str(output_path), image)
    return True


def create_data_yaml(output_path, class_names):
    """
    Create data.yaml file for YOLO format dataset.
    
    Args:
        output_path: Path where to save data.yaml
        class_names: List of class names
    """
    with open(output_path, 'w') as f:
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    
    print(f"✓ Created data.yaml with {len(class_names)} classes")


def convert_masks_to_yolo_bbox(input_dir, output_dir, classes):
    """
    Convert binary masks to YOLO bounding box format.
    
    Args:
        input_dir: Path to directory containing binary masks in subfolders
        output_dir: Path to save YOLO annotations
        classes: Dictionary mapping subfolder names to class IDs
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for class_name, class_id in classes.items():
        mask_folder = os.path.join(input_dir, class_name)
        if not os.path.isdir(mask_folder):
            continue
        
        for mask_file in os.listdir(mask_folder):
            if not mask_file.endswith((".png", ".jpg", ".jpeg")):
                continue
            
            mask_path = os.path.join(mask_folder, mask_file)
            class_name = os.path.basename(mask_folder)
            
            # Remove "_mask" and add class name to output name
            image_name = f"{os.path.splitext(mask_file)[0].replace('_mask', '')}_{class_name}.txt"
            output_path = os.path.join(output_dir, image_name)
            
            # Load binary mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # Convert mask to polygons, then to bounding boxes
            polygons = sv.mask_to_polygons(mask)
            
            annotations = []
            height, width = mask.shape
            for polygon in polygons:
                x_min, y_min, x_max, y_max = sv.polygon_to_xyxy(polygon)
                # Normalize coordinates
                x_min /= width
                y_min /= height
                x_max /= width
                y_max /= height
                scaled_width = (x_max - x_min)
                scaled_height = (y_max - y_min)
                x_center = (x_min + scaled_width / 2)
                y_center = (y_min + scaled_height / 2)
                
                annotation = [class_id, x_center, y_center, scaled_width, scaled_height]
                annotations.append(annotation)
            
            # Write YOLO annotations
            with open(output_path, "w") as f:
                for annotation in annotations:
                    annotation_str = " ".join(map(str, annotation))
                    f.write(annotation_str + "\n")


def copy_and_rename_images(source_dir, output_dir, append_folder_name=True, target_format='jpg'):
    """
    Copy images from nested folder structure to flat structure.
    
    Args:
        source_dir: Source directory with subfolders
        output_dir: Output directory for flat structure
        append_folder_name: If True, append folder name to filename
        target_format: Target image format
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    for parent_folder in os.listdir(source_dir):
        parent_path = os.path.join(source_dir, parent_folder)
        if not os.path.isdir(parent_path):
            continue
        
        for image_file in os.listdir(parent_path):
            if not image_file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            
            image_path = os.path.join(parent_path, image_file)
            image_name = os.path.splitext(image_file)[0]
            
            if append_folder_name:
                new_name = f"{image_name}_{parent_folder}.{target_format}"
            else:
                new_name = f"{image_name}.{target_format}"
            
            output_path = os.path.join(output_dir, new_name)
            
            image = cv2.imread(image_path)
            if image is not None:
                cv2.imwrite(output_path, image)
                count += 1
    
    return count


def create_empty_annotations(image_dir, output_dir, suffix=''):
    """
    Create empty annotation files for images (e.g., for 'good' images with no defects).
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save empty annotation files
        suffix: Optional suffix to add to annotation filenames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    for image_file in os.listdir(image_dir):
        if not image_file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        
        base_name = os.path.splitext(image_file)[0]
        if suffix:
            annotation_name = f"{base_name}_{suffix}.txt"
        else:
            annotation_name = f"{base_name}.txt"
        
        annotation_path = os.path.join(output_dir, annotation_name)
        
        with open(annotation_path, "w") as f:
            pass
        
        count += 1
    
    return count
