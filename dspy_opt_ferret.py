import dspy
import os
from PIL import Image
import json
import supervision as sv
from utils.check_labels import *
from run_any2 import run_any_args
import argparse
import wandb
from utils.wandb_utils import *
import pandas as pd
import subprocess
import torch
import gc
import numpy as np  # Ensure numpy is imported
from utils.check_labels import set_one_class, check_classes

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def label_images(config: None, gt_dataset: sv.DetectionDataset, prompt: str):
    # Create the arguments
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    args = argparse.Namespace(
        config=config_path,
        section="wood",
        model="Florence",
        tag="default",
        sahi=False,
        reload=False,
        ontology=f"{prompt}: defect",
        wandb=False,
        save_images=False,
    )
    dataset = run_any_args(args)
    dataset = set_one_class(dataset)
    gt_dataset = set_one_class(gt_dataset)
    print(check_classes(dataset))
    print(check_classes(gt_dataset))

    confusion_matrix, acc, map_result = evaluate_detections(dataset, gt_dataset)
    print(f"Accuracy: {acc}")

    gt_class = "defect"
    TP = confusion_matrix[0, 0] / confusion_matrix.sum()
    FP = confusion_matrix[0, 1] / confusion_matrix.sum()
    FN = confusion_matrix[1, 0] / confusion_matrix.sum()
    F1 = 2 * TP / (2 * TP + FP + FN)

    return gt_class, TP, FP, FN, acc[0], F1, dataset


def template_prompt(image_feature, location, region_feature, object_name):
    """Create a text-only prompt since Ferret can't process arrays in text."""
    if isinstance(location, tuple):
        location_str = f"at coordinates {location}"
    else:
        location_str = str(location)
    prompt = (
        f"Considering the region {location_str} of the image, "
        f"would you classify it as a {object_name} category? Respond with only 'yes' or 'no'."
    )
    return prompt


class CheckAndRevisePrompt(dspy.Signature):
    """Signature for checking and revising prompts"""
    input_prompt = dspy.InputField()
    output_class = dspy.OutputField(type=bool)


class CheckDetection(dspy.Signature):
    """
    Signature for checking and revising object detection prompts.
    Attributes:
        input_image (dspy.InputField): The full input image to analyze.
        input_box (dspy.InputField): The bounding box coordinates or description of the region of interest.
        input_region (dspy.InputField): The cropped region of the image to be classified.
        input_object (dspy.InputField): The name of the object category to check for.
        output_bool (dspy.OutputField): Boolean output indicating whether the region contains
                                        the specified object without doubt (True) or not (False).
    """
    input_image = dspy.InputField(description="The full input image to analyze.", type=dspy.Image)
    input_box = dspy.InputField(
        description="The bounding box coordinates or description of the region of interest, in the format (x1, y1, x2, y2)."
    )
    input_region = dspy.InputField(
        description="The cropped region of the image to be classified. Corresponds to the bounding box coordinates.",
        type=dspy.Image,
    )
    input_object = dspy.InputField(description="The name of the object category to check for.")
    output_bool = dspy.OutputField(description="Detection correct? True or False", type=bool)


def filter_detection_dataset(ds: sv.DetectionDataset, threshold: float) -> sv.DetectionDataset:
    """
    Filter a sv.DetectionDataset instance by only retaining detections with confidence above the threshold.
    """
    filtered_data = []
    # Iterate through each sample in the dataset
    for image_path, image, annotation in ds:
        # Filter detections based on confidence threshold.
        valid_indices = [i for i, conf in enumerate(annotation.confidence) if conf > threshold]
        if valid_indices:
            filtered_boxes = annotation.xyxy[valid_indices]
            filtered_conf = np.array(annotation.confidence)[valid_indices]
            # If your annotation includes labels, filter them as well.
            filtered_labels = [annotation.labels[i] for i in valid_indices] if hasattr(annotation, "labels") else None

            # Create a new annotation using the filtered values.
            new_annotation = sv.Detections(xyxy=filtered_boxes, confidence=filtered_conf, labels=filtered_labels)
            filtered_data.append((image_path, image, new_annotation))
    # Construct a new DetectionDataset from the filtered data.
    return sv.DetectionDataset(filtered_data)


def main():
    # Initialize wandb
    wandb.login()
    run = wandb.init(project="dspy")

    # Define and configure the language model
    lm = dspy.LM("ollama/qwen2.5:7b", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)

    gc.collect()
    torch.cuda.empty_cache()

    with open("config.json", "r") as f:
        config = json.load(f)["wood"]
    gt_dataset = load_dataset(
        config["GT_IMAGES_DIRECTORY_PATH"],
        config["GT_ANNOTATIONS_DIRECTORY_PATH"],
        config["GT_DATA_YAML_PATH"],
    )

    check_and_revise_prompt = dspy.Predict(CheckAndRevisePrompt)
    check_detection = dspy.Predict(CheckDetection)
    initial_prompt = "defect"
    gt_class, TP, FP, FN, acc, F1, dataset = label_images(config, gt_dataset, initial_prompt)

    ds_review = dataset
    for image_path, image, annotation in ds_review:
        image = Image.open(image_path)
        boxes = annotation.xyxy
        print("Processing image:", image_path)
        for i, bbox in enumerate(boxes):
            # Convert bbox to a tuple of integers
            box_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            # Crop image based on the current bbox
            cropped_image = image.crop(box_tuple)
            object_name = "defect"
            prompt = template_prompt(np.array(image), box_tuple, np.array(cropped_image), object_name)

            response = check_and_revise_prompt(input_prompt=prompt)
            response2 = check_detection(
                input_image=dspy.Image.from_PIL(image),
                input_box=box_tuple,
                input_region=dspy.Image.from_PIL(cropped_image),
                input_object=object_name,
            )
            print("Prompt review response:", response)
            print("Detection check response:", response2)

            # Adjust detection confidence based on reviewer decision
            if "yes" in str(response).lower():
                annotation.confidence[i] = 1
            else:
                annotation.confidence[i] = 0

    # Filter ds_review based on the updated confidence values
    filtered_ds = filter_detection_dataset(ds_review, threshold=0.5)
    confusion_matrix, acc, map_result = evaluate_detections(filtered_ds, gt_dataset)
    print(f"Accuracy after filtering: {acc}")
    print(f"Confusion Matrix after filtering: {confusion_matrix}")


if __name__ == "__main__":
    main()
