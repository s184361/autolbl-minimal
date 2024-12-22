import os
import cv2
from utils.check_labels import load_dataset, compare_plot
import supervision as sv
from ultralytics.data.converter import (
    convert_segment_masks_to_yolo_seg as convert_masks_to_yolo,
)


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
            
            image_name = f"{os.path.splitext(mask_file)[0].replace('_mask', '')}_{class_name}.txt"  # Remove "_mask" and add class name to the output name
            output_path = os.path.join(output_dir, image_name)

            # Load binary mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                continue

            # Use supervision to convert mask to polygons
            polygons = sv.mask_to_polygons(mask)

            annotations = []
            x_min, y_min, x_max, y_max = sv.polygon_to_xyxy(polygons)
                
            annotation = [class_id] + flattened_polygon
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

if __name__ == "__main__":
    # Define the input folder structure and output path
    input_directory = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/ground_truth"
    output_directory = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/yolo_annotations"

    # Define class mappings
    class_mappings = {"broken_large": 0, "broken_small": 1, "contamination": 2}

    # Convert masks to YOLO format
    convert_masks_to_yolo(input_directory, output_directory, class_mappings)

    # Save test images with parent folder names
    test_directory = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/test"
    images_output_directory = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/images"

    save_test_images_with_parent_name(test_directory, images_output_directory)
    annotations_for_good_pictures(test_directory, output_directory)
    yaml_dir = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/bottle/data.yaml"
    dataset = load_dataset(images_directory_path= images_output_directory,
                    annotations_directory_path= output_directory,
                    data_yaml_path=yaml_dir)

    compare_plot(dataset,dataset)
