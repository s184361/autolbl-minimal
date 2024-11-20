import supervision as sv
import os
import numpy as np
import re
import yaml
import matplotlib.pyplot as plt
from autodistill.utils import plot
import cv2
from typing import List

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
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = abs(x1 - x2)
                    height = abs(y1 - y2)
                    new_line = f"{label_number} {center_x} {center_y} {width} {height}\n"
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
    image_dataset = [os.path.splitext(os.path.basename(path))[0] for path, _, _ in dataset]
    image_gt_dataset = [os.path.splitext(os.path.basename(path))[0] for path, _, _ in gt_dataset]
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
            dataset.annotations[key].confidence = np.ones_like(dataset.annotations[key].class_id)
    for key in gt_dataset.annotations.keys():
        for i in range(len(gt_dataset.annotations[key])):
            gt_dataset.annotations[key].confidence = np.ones_like(gt_dataset.annotations[key].class_id)
    #load confidence from labels confidence-annotation.txt
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
    gt_dict = {os.path.basename(image_path).replace(".png", ".jpg"): annotation for image_path, _, annotation in gt_dataset}
    for image_path, _, annotation in dataset:
        key = os.path.basename(image_path)
        predictions.append(annotation)
        if key in gt_dict:
            targets.append(gt_dict[key])
    confusion_matrix = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=dataset.classes,
        iou_threshold=0.5
    )
    fig = confusion_matrix.plot(normalize=True)
    plt.show()
    plt.savefig("results/confusion_matrix.png")
    print(confusion_matrix)
    sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets
    )

def plot(image: np.ndarray, detections, classes: List[str], raw=False):
    """
    Plot bounding boxes or segmentation masks on an image.

    Args:
        image: The image to plot on
        detections: The detections to plot
        classes: The classes to plot
        raw: Whether to return the raw image or plot it interactively

    Returns:
        The raw image (np.ndarray) if raw=True, otherwise None (image is plotted interactively
    """
    # TODO: When we have a classification annotator
    # in supervision, we can add it here
    if detections.mask is not None:
        annotator = sv.MaskAnnotator()
    else:
        #annotator = sv.BoundingBoxAnnotator()
        annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _, _ in detections
    ]

    annotated_frame = annotator.annotate(scene=image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, labels=labels, detections=detections
    )

    if raw:
        return annotated_frame

    sv.plot_image(annotated_frame)
def compare_plot(dataset,gt_dataset):
    img = []
    name = []
    for image_path, _, annotation in dataset:
        image = cv2.imread(image_path)
        classes = dataset.classes
        result = annotation

        img.append(plot(
        image=image,
        classes=classes,
        detections=result,
        raw=True
        ))
        name.append(os.path.basename(image_path))
    
    for image_path, _, annotation in gt_dataset:
        classes = gt_dataset.classes
        name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        if name_gt in name:
            image = cv2.imread(image_path)
            result = annotation
            #check if annotation is empty
            if len(result) == 0:
                #if empty, plot only the image
                img_gt = image
            else:
                img_gt = plot(
                image=image,
                classes=classes,
                detections=result,
                raw=True
                )

            # Find fig index
            index = name.index(name_gt)
            fig = plt.figure()
            fig.add_subplot(2, 1, 1)
            plt.imshow(img[index])
            plt.title("Inference")
            plt.axis('off')
            fig.add_subplot(2, 1, 2)
            plt.imshow(img_gt)
            plt.title("Ground Truth")
            plt.axis('off')

            # Add spacing between figures
            plt.subplots_adjust(hspace=0.5)

            # Save in high resolution
            plt.savefig(f"results/{name_gt}", dpi=600)
            plt.close(fig)


    

def main():
    HOME = os.getcwd()
    ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
    IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
    DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"
    GT_ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/data/BoudingBoxes"
    GT_IMAGES_DIRECTORY_PATH = f"{HOME}/images"
    GT_DATA_YAML_PATH = f"{HOME}/data/data.yaml"

    print_supervision_version()
    dataset = load_dataset(IMAGES_DIRECTORY_PATH, ANNOTATIONS_DIRECTORY_PATH, DATA_YAML_PATH)
    update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
    gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
    #compare_classes(gt_dataset, dataset)
    #compare_image_keys(gt_dataset, dataset)
    evaluate_detections(dataset, gt_dataset)
    compare_plot(dataset,gt_dataset)

if __name__ == "__main__":
    main()