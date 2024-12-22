import supervision as sv
import os
import numpy as np
import re
import yaml
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import shutil  # Import shutil for file operations
from autodistill.utils import plot
import wandb
import shutil

def print_supervision_version():
    print("Supervision version:", sv.__version__)


def load_dataset(images_directory_path, annotations_directory_path, data_yaml_path):
    return sv.DetectionDataset.from_yolo(
        images_directory_path=images_directory_path,
        annotations_directory_path=annotations_directory_path,
        data_yaml_path=data_yaml_path,
    )


def update_labels(gt_annotations_directory_path, gt_data_yaml_path):
    with open(gt_data_yaml_path, "r") as file:
        data_yaml = yaml.safe_load(file)
    label_map = {label: idx for idx, label in enumerate(data_yaml["names"])}
    labels_directory = os.path.join(gt_annotations_directory_path)

    for filename in os.listdir(labels_directory):
        if "_anno.txt" in filename:
            file_path = os.path.join(labels_directory, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    label = parts[0].lower().replace("_", " ")
                    x1, y1, x2, y2 = map(float, parts[1:])
                    label_number = label_map.get(label)
                    if label_number is None:
                        print(label,label_map)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = abs(x1 - x2)
                    height = abs(y1 - y2)
                    new_line = (
                        f"{label_number} {center_x} {center_y} {width} {height}\n"
                    )
                    new_lines.append(new_line)
            new_filename = re.sub(r"_anno", "", filename)
            new_file_path = os.path.join(labels_directory, new_filename)
            with open(new_file_path, "w") as file:
                file.writelines(new_lines)


def compare_classes(gt_dataset, dataset):
    print("GT classes: ", gt_dataset.classes)
    print("Dataset classes: ", dataset.classes)
    gt_classes_not_in_dataset = set(gt_dataset.classes) - set(dataset.classes)
    print("GT classes not in dataset classes: ", gt_classes_not_in_dataset)


def compare_image_keys(gt_dataset, dataset):
    image_dataset = [
        os.path.splitext(os.path.basename(path))[0] for path, _, _ in dataset
    ]
    image_gt_dataset = [
        os.path.splitext(os.path.basename(path))[0] for path, _, _ in gt_dataset
    ]
    image_dataset.sort()
    image_gt_dataset.sort()
    print("Dataset images: ", image_dataset)
    print("GT images: ", image_gt_dataset)
    print("GT images not in dataset images: ", image_gt_dataset == image_dataset)
    for path, _, _ in dataset:
        key = os.path.basename(path)
        if os.path.splitext(key)[0] not in image_gt_dataset:
            print(f"Key {key} not in GT dataset")
        else:
            print(f"Key {key} in GT dataset")


def evaluate_detections(dataset, gt_dataset):

    for key in dataset.annotations.keys():
        for i in range(len(dataset.annotations[key])):
            dataset.annotations[key].confidence = np.ones_like(
                dataset.annotations[key].class_id
            )
    for key in gt_dataset.annotations.keys():
        for i in range(len(gt_dataset.annotations[key])):
            gt_dataset.annotations[key].confidence = np.ones_like(
                gt_dataset.annotations[key].class_id
            )
    # load confidence from labels confidence-annotation.txt
    """ for image_path, _, annotation in dataset:
        key = os.path.basename(image_path)
        directory_path = os.path.dirname(image_path)
        base_path, _ = os.path.split(directory_path)
        base_path, _ = os.path.split(base_path)
        with open(f"{base_path}/labels/confidence-{key.replace('.jpg', '.txt')}") as f:
            lines = f.readlines()
        for i in range(len(annotation)):
            annotation[i].confidence = float(lines[i].split()[1]) """
    predictions = []
    targets = []
    gt_dict = {
        os.path.basename(image_path).replace(".png", ".jpg"): annotation
        for image_path, _, annotation in gt_dataset
    }
    for image_path, _, annotation in dataset:
        key = os.path.basename(image_path)
        predictions.append(annotation)
        if key in gt_dict:
            targets.append(gt_dict[key])
    confusion_matrix = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=dataset.classes,
        iou_threshold=0.5,
    )
    fig =confusion_matrix.plot(normalize=True)
    try:
        wandb.log({"Confusion Matrix": wandb.Image(fig)})
    except NameError:
        print("WandB not available")
    plt.savefig("results/confusion_matrix.png")
    print(confusion_matrix)
    map_metric = sv.metrics.MeanAveragePrecision()
    map_result = map_metric.update(predictions, targets).compute()
    # table = wandb.Table(data=map_result.mAP_scores)
    # wandb.log({"mAP Results": table})
    map_result.plot()
    fig = plt.gcf()  # grab last figure
    try:
        wandb.log({"mAP": wandb.Image(fig)})
    except NameError:
        print("WandB not available")
    plt.savefig("results/mAP.png")


def compare_plot(dataset, gt_dataset):
    img = []
    name = []
    for image_path, _, annotation in dataset:
        image = cv2.imread(image_path)
        classes = dataset.classes
        result = annotation

        img.append(plot(image=image, classes=classes, detections=result, raw=True))
        name.append(os.path.basename(image_path))

    for image_path, _, annotation in gt_dataset:
        classes = gt_dataset.classes
        name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        if name_gt in name:
            image = cv2.imread(image_path)
            result = annotation
            # check if annotation is empty
            if len(result) == 0:
                # if empty, plot only the image
                img_gt = image
            else:
                img_gt = plot(image=image, classes=classes, detections=result, raw=True)

            # Find fig index
            index = name.index(name_gt)
            fig = plt.figure()
            fig.add_subplot(2, 1, 1)
            plt.imshow(img[index])
            plt.title("Inference")
            plt.axis("off")
            fig.add_subplot(2, 1, 2)
            plt.imshow(img_gt)
            plt.title("Ground Truth")
            plt.axis("off")

            # Add spacing between figures
            plt.subplots_adjust(hspace=0.5)

            try:
                wandb.log({f"Annotated Image {name_gt}": wandb.Image(fig)})
            except:
                print("WandB not available")
            plt.savefig(f"results/{name_gt}", dpi=600)
            plt.close(fig)


def find_single_annotation_files(gt_annotations_directory_path, gt_data_yaml_path):
    """
    Find annotation files with only one annotation and return one file per class.

    Args:
        gt_annotations_directory_path: Path to ground truth annotations directory
        gt_data_yaml_path: Path to YAML file containing class names

    Returns:
        dict: Dictionary with class names as keys and corresponding filenames as values
    """
    # Load class names from YAML
    with open(gt_data_yaml_path, "r") as file:
        data_yaml = yaml.safe_load(file)
    class_names = data_yaml["names"]

    # Initialize dictionary to store results
    single_annotations_per_class = {class_name: None for class_name in class_names}

    # Iterate through annotation files
    for filename in os.listdir(gt_annotations_directory_path):
        if filename.endswith(".txt") and not filename.startswith("confidence-"):
            file_path = os.path.join(gt_annotations_directory_path, filename)

            # Read the file
            with open(file_path, "r") as file:
                lines = file.readlines()

                # Check if file has exactly one annotation
                if len(lines) == 1:
                    # Parse the class name and convert to index
                    parts = lines[0].strip().split()
                    if (
                        len(parts) == 5
                    ):  # Ensure we have all 5 parts (class + 4 coordinates)
                        class_name = parts[0].lower().replace("_", " ")
                        try:
                            class_id = class_names.index(class_name)
                            # If we haven't found an example for this class yet, store it
                            if (
                                single_annotations_per_class[class_names[class_id]]
                                is None
                            ):
                                single_annotations_per_class[class_names[class_id]] = (
                                    filename
                                )
                        except ValueError:
                            # Skip if class name not found in class_names
                            continue

    # Filter out classes with no single-annotation examples
    result = {
        class_name: filename
        for class_name, filename in single_annotations_per_class.items()
        if filename is not None
    }

    return result


def create_annotations_dataframe(annotations_directory_path, gt_data_yaml_path):
    """
    Create a pandas DataFrame of annotations that can be easily queried.

    Args:
        annotations_directory_path: Path to annotations directory
        gt_data_yaml_path: Path to YAML file containing class names

    Returns:
        pandas DataFrame containing all annotations
    """
    # Load class names from YAML
    with open(gt_data_yaml_path, "r") as file:
        data_yaml = yaml.safe_load(file)
    class_names = data_yaml["names"]

    # Initialize list to store annotation data
    data = []

    # Iterate through annotation files
    for filename in os.listdir(annotations_directory_path):
        if filename.endswith(".txt") and not filename.startswith("confidence-"):
            file_path = os.path.join(annotations_directory_path, filename)

            # Read the file
            with open(file_path, "r") as file:
                lines = file.readlines()

                # Process each annotation line
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # Ensure we have all 5 parts
                        class_name = parts[0].lower().replace("_", " ")
                        try:
                            class_id = class_names.index(class_name)
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            data.append(
                                {
                                    "filename": filename,
                                    "class_name": class_names[class_id],
                                    "class_id": class_id,
                                    "x_center": x_center,
                                    "y_center": y_center,
                                    "width": width,
                                    "height": height,
                                }
                            )
                        except ValueError:
                            # Skip if class name not found in class_names
                            print(class_name,class_names)
                            continue

    # Create DataFrame from collected data
    df = pd.DataFrame(data)
    return df


def query_annotations(df, class_name=None, filename=None):
    """
    Query the annotations DataFrame.

    Args:
        df: pandas DataFrame containing annotations
        class_name: Filter by class name
        filename: Filter by filename

    Returns:
        Filtered pandas DataFrame
    """
    mask = pd.Series([True] * len(df))

    if class_name:
        mask &= df["class_name"] == class_name

    if filename:
        mask &= df["filename"] == filename

    return df[mask]


def save_annotations(df, csv_path="annotations.csv"):
    """
    Save annotations DataFrame to CSV file.

    Args:
        df: pandas DataFrame containing annotations
        csv_path: Path where to save the CSV file
    """
    df.to_csv(csv_path, index=False)


def load_annotations(csv_path="annotations.csv"):
    """
    Load annotations from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        pandas DataFrame containing annotations
    """
    return pd.read_csv(csv_path)


def print_statistics(df):
    """
    Print various statistics about the annotations.

    Args:
        df: pandas DataFrame containing annotations
    """
    print("\nDataset Statistics:")
    print("-" * 50)

    # Count of annotations per class
    print("\nClass distribution:")
    print(df["class_name"].value_counts())

    # Count of annotations per file
    print("\nAnnotations per file:")
    print(df.groupby("filename").size())

    # Basic statistics of bounding box dimensions
    print("\nBounding box statistics:")
    print(df[["width", "height"]].describe())


def convert_bmp_to_jpg(image_dir_path):
    for root, dirs, files in os.walk(image_dir_path):
        for file in files:
            if file.endswith(".bmp"):
                img = cv2.imread(os.path.join(root, file))
                cv2.imwrite(os.path.join(root, file.replace(".bmp", ".jpg")), img)


def load_images_and_annotations(images_dir, annotations_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [
        f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)

        # Load corresponding annotation file
        annotation_file = os.path.splitext(image_file)[0] + ".txt"
        annotation_path = os.path.join(annotations_dir, annotation_file)

        if not os.path.exists(annotation_path):
            print(
                f"Warning: Annotation file {annotation_path} does not exist. Skipping {image_file}."
            )
            continue

        with open(annotation_path, "r") as file:
            annotations = file.readlines()

        # Process each annotation
        for idx, line in enumerate(annotations):
            parts = line.strip().split()
            if len(parts) != 5:
                print(
                    f"Warning: Invalid annotation format in {annotation_file} on line {idx + 1}."
                )
                continue

            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert normalized coordinates to pixel values
            img_height, img_width = image.shape[:2]
            x_center_pixel = int(x_center * img_width)
            y_center_pixel = int(y_center * img_height)
            width_pixel = int(width * img_width)
            height_pixel = int(height * img_height)

            # Calculate bounding box coordinates
            x1 = int(x_center_pixel - (width_pixel / 2))
            y1 = int(y_center_pixel - (height_pixel / 2))
            x2 = int(x_center_pixel + (width_pixel / 2))
            y2 = int(y_center_pixel + (height_pixel / 2))

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]

            # Save the cropped image
            # output_file_name = f"{os.path.splitext(image_file)[0]}_{class_id}.jpg"
            # use yaml file class_id keys
            keys = [
                "live knot",
                "dead knot",
                "knot missing",
                "knot with crack",
                "crack",
                "quartzity",
                "resin",
                "marrow",
                "blue stain",
                "overgrown"
            ]
            output_file_name = f"{keys[int(class_id)]}.jpg"
            output_path = os.path.join(output_dir, output_file_name)
            cv2.imwrite(output_path, cropped_image)

        print(f"Processed {image_file}, saved cropped images to {output_dir}.")


def reset_folders(dataset_folder_path, results_folder_path):
    """
    Deletes the dataset folder and results folder, then creates a new results folder.

    Args:
        dataset_folder_path: Path to the dataset folder to be deleted.
        results_folder_path: Path to the results folder to be deleted and recreated.
    """
    # Delete the dataset folder if it exists
    if os.path.exists(dataset_folder_path):
        shutil.rmtree(dataset_folder_path)
        print(f"Deleted dataset folder: {dataset_folder_path}")

    # Delete the results folder if it exists
    if os.path.exists(results_folder_path):
        shutil.rmtree(results_folder_path)
        print(f"Deleted results folder: {results_folder_path}")

    # Create a new results folder
    os.makedirs(results_folder_path, exist_ok=True)
    print(f"Created new results folder: {results_folder_path}")

def plot_annotated_images(dataset, sample_size, save_path):
    from utils.config import SAMPLE_GRID_SIZE, SAMPLE_PLOT_SIZE
    image_names = list(dataset.images.keys())[:sample_size]

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()

    images = []
    for image_name in image_names:
        image = dataset.images[image_name]
        annotations = dataset.annotations[image_name]
        labels = [dataset.classes[class_id] for class_id in annotations.class_id]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=annotations)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations)
        images.append(annotated_image)
        sv.plot_images_grid(
                images=images,
                titles=image_names,
                grid_size=SAMPLE_GRID_SIZE,
                size=SAMPLE_PLOT_SIZE,
            )
        plt.axis("off")
        fig = plt.gcf()
    # Log the combined grid of annotated images to wandb
    try:
        print("modify back")
        #wandb.log({"Annotated Images Grid": [wandb.Image(fig)]})
    except NameError:
        # Save the images to the specified save path if wandb is not available
        plt.savefig(save_path, dpi=1200)
        print(f"Saved annotated images grid to {save_path}.")

def main():
    wandb.init()
    print_supervision_version()
    dataset = load_dataset(
        IMAGES_DIRECTORY_PATH, ANNOTATIONS_DIRECTORY_PATH, DATA_YAML_PATH
    )
    update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
    gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
    compare_classes(gt_dataset, dataset)
    compare_image_keys(gt_dataset, dataset)
    evaluate_detections(dataset, gt_dataset)
    compare_plot(dataset,gt_dataset)
    #load_images_and_annotations(f"{HOME}/Image_Embeddings", GT_ANNOTATIONS_DIRECTORY_PATH, f"{HOME}/croped_images")
    # single_annotation_files = find_single_annotation_files(
    #    GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH
    # )
    # print(single_annotation_files)
    # df = create_annotations_dataframe(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
    # print(df)
    # print_statistics(df)
    # save_annotations(df)
    # loaded_df = load_annotations()
    # print(loaded_df)


if __name__ == "__main__":
    from config import *

    main()
    if wandb.run is not None:
        wandb.finish()
