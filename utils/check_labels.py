import supervision as sv
import os
import numpy as np
import re
import yaml
import matplotlib
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import shutil  # Import shutil for file operations
from autodistill.utils import plot
import wandb
import shutil
from typing import Callable, List, Optional, Tuple
import subprocess
from joblib import Parallel, delayed
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
                    parts[1:] = [part.replace(",", ".") for part in parts[1:]]
                    x1, y1, x2, y2 = map(float, parts[1:])
                    label_number = label_map.get(label)
                    if label_number is None:
                        print(label,label_map)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = abs(x1 - x2)
                    height = abs(y1 - y2)
                    #check if annotation is smaller than 0.1% of the image
                    if width*height > 0.1:
                        pass
                        print(f"annotation skipped for image {filename}")
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

def evaluate_detections(dataset, gt_dataset, results_dir="results"):
    # Set confidence for all annotations in both datasets
    def set_confidence(annotations):
        for key in annotations.keys():
            for i in range(len(annotations[key])):
                annotations[key][i].confidence = np.ones_like(annotations[key][i].class_id)

    set_confidence(dataset.annotations)
    set_confidence(gt_dataset.annotations)

    # Create ground truth dictionary
    gt_dict = {
        os.path.basename(image_path).replace(".png", ".jpg"): annotation
        for image_path, _, annotation in gt_dataset
    }

    # Prepare predictions and targets
    predictions = []
    targets = []

    for image_path, _, annotation in dataset:
        key = os.path.basename(image_path)
        annotation.confidence = np.ones_like(annotation.class_id)
        predictions.append(annotation)
        if key in gt_dict:
            annotation = gt_dict[key]
            annotation.confidence = np.ones_like(annotation.class_id)
            targets.append(gt_dict[key])

    # Compute confusion matrix
    if isinstance(dataset, sv.ClassificationDataset):
        confusion_matrix = np.zeros((len(gt_dataset.classes), len(dataset.classes)))
        for target, pred in zip(targets, predictions):
            confusion_matrix[int(target.class_id), int(pred.class_id)] += 1
        fig = plot_confusion_class(
            input=confusion_matrix,
            classes=gt_dataset.classes,
            normalize=True
        )
    else:
        confusion_matrix = sv.ConfusionMatrix.from_detections(
            predictions=predictions,
            targets=targets,
            classes=dataset.classes,
            iou_threshold=0.5,
        )
        fig = confusion_matrix.plot(normalize=True)
        #if there is more than one class
        if len(dataset.classes) > 1:
            confusion_matrix = confusion_matrix.matrix[:-1, :-1]  # Remove last row and column
        else:
            confusion_matrix = confusion_matrix.matrix

    acc = confusion_matrix.diagonal() / confusion_matrix.sum(-1)
    acc = np.append(acc, confusion_matrix.diagonal().sum() / confusion_matrix.sum())
    print("acc", acc)

    try:
        wandb.log({"Confusion Matrix": wandb.Image(fig)})
        tab = wandb.Table(columns=gt_dataset.classes + ["all"], data=[acc])
        wandb.log({"Accuracies": tab})
    except Exception as e:
        print(f"WandB logging error: {e}")

    plt.savefig(f"{results_dir}/confusion_matrix.png")
    print(confusion_matrix)

    # Compute mAP if both datasets are DetectionDataset
    if isinstance(dataset, sv.DetectionDataset) and isinstance(gt_dataset, sv.DetectionDataset):
        map_metric = sv.metrics.MeanAveragePrecision()
        map_result = map_metric.update(predictions, targets).compute()
        print(map_result)

        map_result.plot()
        fig = plt.gcf()  # grab last figure
        try:
            wandb.log({"mAP": wandb.Image(fig)})
        except Exception as e:
            print(f"WandB logging error: {e}")
        plt.savefig(f"{results_dir}/mAP.png")
    return confusion_matrix, acc, map_result

def compare_plot(dataset, gt_dataset, results_dir="results"):
    # Ensure confidence is set for all annotations in both datasets
    for key in dataset.annotations.keys():
        for i in range(len(dataset.annotations[key])):
            dataset.annotations[key][i].confidence = np.ones_like(
                dataset.annotations[key][i].class_id
            )
    for key in gt_dataset.annotations.keys():
        for i in range(len(gt_dataset.annotations[key])):
            gt_dataset.annotations[key][i].confidence = np.ones_like(
                gt_dataset.annotations[key][i].class_id
            )

    img = []
    name = []

    # Process dataset images
    for image_path, _, annotation in dataset:
        image = cv2.imread(image_path)
        classes = dataset.classes
        result = annotation
        try:
            img.append(plot(image=image, classes=classes, detections=result, raw=True))
        except Exception as e:
            print(f"Error plotting inference image: {e}")
            img.append(plot(image=image, classes=[str(i) for i in range(100)], detections=result, raw=True))
        name.append(os.path.basename(image_path))

    # Process ground truth images
    for image_path, _, annotation in gt_dataset:
        classes = gt_dataset.classes
        name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        if name_gt in name:
            image = cv2.imread(image_path)
            result = annotation
            if len(result) == 0:
                img_gt = image
            else:
                try:
                    if result.confidence is None:
                        result.confidence = np.ones_like(result.class_id)
                    img_gt = plot(image=image, classes=classes, detections=result, raw=True)
                except Exception as e:
                    print(f"Error plotting ground truth image: {e}")
                    img_gt = plot(image=image, classes=[str(i) for i in range(100)], detections=result, raw=True)

            # Find fig index
            index = name.index(name_gt)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
            axes[0].imshow(img[index])
            axes[0].set_title("Inference")
            axes[0].axis("off")
            axes[1].imshow(img_gt)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            fig.patch.set_facecolor('none')

            try:
                wandb.log({f"Annotated Image {name_gt}": wandb.Image(fig)})
            except Exception as e:
                print(f"WandB logging error: {e}")
            plt.savefig(os.path.join(results_dir, name_gt), dpi=1200)
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


def convert_bmp_to_jpg(image_dir_path,delete_bmp=False):
    for root, dirs, files in os.walk(image_dir_path):
        for file in files:
            if file.endswith(".bmp"):
                img = cv2.imread(os.path.join(root, file))
                cv2.imwrite(os.path.join(root, file.replace(".bmp", ".jpg")), img)
                if delete_bmp:
                    os.remove(os.path.join(root, file))


def crop_gt_images(images_dir, annotations_dir, output_dir, keys=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [
        f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    for image_file in image_files:
        # Load the image
        image_name = os.path.splitext(image_file)[0]
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
            if width_pixel < 50 or height_pixel<50:
                continue
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

            # use yaml file class_id keys
            if keys is None:
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
            # output_file_name = f"{image_file}_{keys[int(class_id)]}.jpg"

            # Save the cropped image
            output_file_name = (
                f"{os.path.splitext(image_file)[0]}_{keys[int(class_id)]}.jpg"
            )
            output_path = os.path.join(output_dir, output_file_name)
            cv2.imwrite(output_path, cropped_image)
            # create annotation files
            annotation = [int(class_id), 0.5, 0.5, 1, 1]
            # Write YOLO annotations to a text file
            output_path = os.path.join(
                output_dir,
                f"{os.path.splitext(image_file)[0]}_{keys[int(class_id)]}.txt",
            )

            with open(output_path, "w") as f:
                annotation_str = " ".join(map(str, annotation))
                f.write(annotation_str)

        print(f"Processed {image_file}, saved cropped images to {output_dir}.")
def create_boundingboxes_defect_annotations(input_dir, output_dir):
    """
    Create a new directory with BoundingBox annotations where the first number in each line is switched to 1.

    Args:
        input_dir: Path to the input directory containing original BoundingBox annotations.
        output_dir: Path to the output directory where modified annotations will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            with open(input_file_path, "r") as infile, open(output_file_path, "w") as outfile:
                lines = infile.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        parts[0] = "0"  # Switch the first number to 0
                        outfile.write(" ".join(parts) + "\n")

    print(f"Processed annotations saved to {output_dir}")

def reset_folders(dataset_folder_path, results_folder_path):
    """
    Deletes the dataset folder and results folder, then creates a new results folder.

    Args:
        dataset_folder_path: Path to the dataset folder to be deleted.
        results_folder_path: Path to the results folder to be deleted and recreated.
    """
    # Delete the dataset folder if it exists
    if os.path.exists(dataset_folder_path):
        try:
            shutil.rmtree(dataset_folder_path)
        except:
            subprocess.run(["rd", "/s", "/q", dataset_folder_path], shell=True, check=True)
        print(f"Deleted dataset folder: {dataset_folder_path}")

    # Delete the results folder if it exists
    if os.path.exists(results_folder_path):
        try:
            shutil.rmtree(results_folder_path)
        except:
            subprocess.run(["rd", "/s", "/q", results_folder_path], shell=True, check=True)
        print(f"Deleted results folder: {results_folder_path}")

    # Create a new results folder
    os.makedirs(results_folder_path, exist_ok=True)
    print(f"Created new results folder: {results_folder_path}")


def process_image(image, annotations, mask_annotator, box_annotator):
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=annotations)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations)
    return annotated_image

def plot_annotated_images(dataset, sample_size, save_path):
    from utils.config import SAMPLE_GRID_SIZE, SAMPLE_PLOT_SIZE
    images =[]
    image_names = []
    annotations = []
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    for image_path, img, annotation in dataset:
        image_names.append(os.path.basename(image_path))
        images.append(process_image(img, annotation, mask_annotator, box_annotator))
        if len(images) == sample_size:
            break

    #free up memory
    del dataset

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
        wandb.log({"Annotated Images Grid": [wandb.Image(fig)]})
    except:
        # Save the images to the specified save path if wandb is not available
        print("WandB not available")
    plt.savefig(save_path, dpi=1200)
    print(f"Saved annotated images grid to {save_path}.")

def plot_confusion_class(
        input,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        classes: Optional[List[str]] = None,
        normalize: bool = False,
        fig_size: Tuple[int, int] = (12, 10),
    ) -> matplotlib.figure.Figure:
    array = input.copy()

    if normalize:
        eps = 1e-8
        array = array / (array.sum(0).reshape(1, -1) + eps)

    array[array < 0.005] = np.nan

    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True, facecolor="white")

    class_names = classes
    use_labels_for_ticks = class_names is not None and (0 < len(class_names) < 99)
    if use_labels_for_ticks:
        x_tick_labels = [*class_names, "FN"]
        y_tick_labels = [*class_names, "FP"]
        num_ticks = len(x_tick_labels)
    else:
        x_tick_labels = None
        y_tick_labels = None
        num_ticks = len(array)
    im = ax.imshow(array, cmap="Blues")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.mappable.set_clim(vmin=0, vmax=np.nanmax(array))

    if x_tick_labels is None:
        tick_interval = 2
    else:
        tick_interval = 1
    ax.set_xticks(np.arange(0, num_ticks, tick_interval), labels=x_tick_labels)
    ax.set_yticks(np.arange(0, num_ticks, tick_interval), labels=y_tick_labels)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="default")

    labelsize = 10 if num_ticks < 50 else 8
    ax.tick_params(axis="both", which="both", labelsize=labelsize)

    if num_ticks < 30:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                n_preds = array[i, j]
                if not np.isnan(n_preds):
                    ax.text(
                        j,
                        i,
                        f"{n_preds:.2f}" if normalize else f"{n_preds:.0f}",
                        ha="center",
                        va="center",
                        color="black" if n_preds < 0.5 * np.nanmax(array) else "white",
                    )

    if title:
        ax.set_title(title, fontsize=20)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_facecolor("white")
    if save_path:
        fig.savefig(
                save_path, dpi=250, facecolor=fig.get_facecolor(), transparent=True
            )
    return fig

def summarize_annotation_distributions(gt_dataset):
    """
    Summarizes the distributions of annotations per image and the distribution of classes among all annotations.

    Args:
        gt_dataset: Ground truth dataset containing images and annotations.

    Returns:
        dict: Dictionary containing summary statistics.
    """
    annotation_counts_per_image = []
    class_distribution = {class_name: 0 for class_name in gt_dataset.classes}

    for _, _, annotations in gt_dataset:
        annotation_counts_per_image.append(len(annotations))
        for class_id in annotations.class_id:
            class_name = gt_dataset.classes[class_id]
            class_distribution[class_name] += 1

    summary = {
        "annotations_per_image": {
            "mean": np.mean(annotation_counts_per_image),
            "std": np.std(annotation_counts_per_image),
            "min": np.min(annotation_counts_per_image),
            "max": np.max(annotation_counts_per_image),
        },
        "class_distribution": class_distribution,
    }
    try:
        wandb.log({"Annotation Summary": wandb.Table(dataframe=pd.DataFrame.from_dict(summary["class_distribution"], orient='index', columns=['count']))})
    except:
        print("WandB not available")
    print("Annotations per image:")
    print(f"Mean: {summary['annotations_per_image']['mean']:.2f}")
    print(f"Std: {summary['annotations_per_image']['std']:.2f}")
    print(f"Min: {summary['annotations_per_image']['min']}")
    print(f"Max: {summary['annotations_per_image']['max']}")
    print("\nClass distribution:")
    for class_name, count in summary["class_distribution"].items():
        print(f"{class_name}: {count}")
    return summary
def classificaiton_table(
        dataset,
        gt_dataset):
    "creates wandb table"
def main():
    import config as config
    """
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
    load_images_and_annotations(
    #    IMAGE_DIR_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, f"{HOME}/croped_images2"
    #)
    """
    #convert_bmp_to_jpg("/work3/s184361/data/Images",delete_bmp=True)
    print(os.path.exists("/work3/s184361/data/BoundingBoxes"))
    #update_labels("/work3/s184361/data/BoudingBoxes", GT_DATA_YAML_PATH)
    keys = [
        "blue stain",
        "crack",
        "dead knot",
        "knot missing",
        "knot with crack",
        "live knot",
        "marrow",
        "overgrown",
        "quartzity",
        "resin"
    ]
    """
    crop_gt_images(
        "/work3/s184361/data/Images",
        "/work3/s184361/data/BoudingBoxes",
        "/work3/s184361/data/croped_images3",
        keys=keys,
    )
    """
    create_boundingboxes_defect_annotations("/zhome/4a/b/137804/Desktop/autolbl/data/BoundingBoxes", "/zhome/4a/b/137804/Desktop/autolbl/data/BoundingBoxes2")
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
