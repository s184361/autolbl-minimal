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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from autodistill.detection import CaptionOntology

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
        f"Considering the region {location_str} of the image {image_feature}, "
        f"would you classify it as a {object_name} category? Return a JSON response with the field 'output_class' set to 'yes' or 'no'."
    )
    return prompt


class CheckAndRevisePrompt(dspy.Signature):
    """Signature for checking and revising prompts"""
    input_prompt = dspy.InputField(description="The prompt describing the image bounding box to be reviewed.")
    output_class = dspy.OutputField(description="Prompt review response", type=str)


class CheckDetection(dspy.Signature):
    """
    Signature for checking and revising object detection prompts.
    Attributes:
        input_image (dspy.InputField): The full input image to analyze.
        input_box (dspy.InputField): The bounding box coordinates or description of the region of interest.
        input_region (dspy.InputField): The cropped region of the image to be classified.
        input_object (dspy.InputField): The name of the object category to check for.
        output_bool (dspy.OutputField): Boolean output indicating whether the region contains
                                        the specified object without doubt (True) or not (False). Output is a JSON response.
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
    output_bool = dspy.OutputField(description="Detection correct? Return a JSON response with the field 'output_bool' set to True or False. You must return True or False, not yes or no or empty string.")


def filter_detection_dataset(ds: sv.DetectionDataset, threshold: float) -> sv.DetectionDataset:
    """
    Filter a sv.DetectionDataset instance by only retaining detections with confidence above the threshold.
    """
    # Extract the classes from the original dataset
    classes = ds.classes
    
    # Create dictionaries for images and annotations
    filtered_images = {}
    filtered_annotations = {}
    
    # Iterate through each sample in the dataset
    for image_path, image, annotation in ds:
        # Filter detections based on confidence threshold
        valid_indices = [i for i, conf in enumerate(annotation.confidence) if conf > threshold]
        if valid_indices:
            filtered_boxes = annotation.xyxy[valid_indices]
            filtered_conf = np.array(annotation.confidence)[valid_indices]
            
            # If your annotation includes labels, filter them as well
            if hasattr(annotation, "labels") and annotation.labels is not None:
                filtered_labels = [annotation.labels[i] for i in valid_indices]
            else:
                # If no labels, use class_id if available, otherwise None
                filtered_labels = annotation.class_id[valid_indices] if hasattr(annotation, "class_id") else None
            
            # Create a new annotation using the filtered values
            new_annotation = sv.Detections(
                xyxy=filtered_boxes, 
                confidence=filtered_conf, 
                class_id=filtered_labels
            )
            
            # Use image path as key for both dictionaries
            key = os.path.basename(image_path)
            filtered_images[key] = image
            filtered_annotations[key] = new_annotation
    
    # Construct a new DetectionDataset with proper parameters
    return sv.DetectionDataset(classes=classes, images=filtered_images, annotations=filtered_annotations)


def process_detection_prompt(check_and_revise_prompt, prompt):
    try:
        response = check_and_revise_prompt(input_prompt=prompt)
        return response
    except Exception as e:
        # If parsing fails, extract yes/no from raw text
        # This is a fallback mechanism
        try:
            raw_response = str(e).lower()
            if "yes" in raw_response:
                return "yes"
            elif "no" in raw_response:
                return "no"
            else:
                return "no"  # Default to "no" if unclear
        except:
            return "no"  # Default to "no" on any error

def log_wandb_metrics(metrics):
    """
    Log metrics to Weights & Biases.
    Args:
        metrics (dict): Dictionary of metrics to log.
    """
    wandb.log(metrics)

def create_review_prompt(box_tuple, object_name):
    """
    Create a prompt for Qwen to review a detection.
    """
    return f"Examine this image. Does it contain a {object_name}? Answer with only 'true' or 'false'."

def review_detection_with_qwen(model, processor, image, box_tuple, cropped_image, object_name):
    """
    Uses Qwen2.5-VL to review a detection directly through transformers.
    
    Args:
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        image: PIL image of the full scene
        box_tuple: Coordinates of the bounding box
        cropped_image: Cropped region of the image
        object_name: Name of the object to check for
    
    Returns:
        Boolean indicating if detection is valid
    """
    # Create prompt message with image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": cropped_image,
                },
                {
                    "type": "text",
                    "text": "Please look at this cropped image and wait for the next message.",
                },
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text", 
                    "text": create_review_prompt(box_tuple, object_name)
                },
            ],
        }
    ]
    
    try:
        # Process input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to appropriate device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Raw Qwen response: {response}")
        
        # Parse response - looking for true/false
        response_lower = response.lower()
        if "true" in response_lower or "yes" in response_lower:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error in Qwen review: {e}")
        return False  # Default to rejecting on error


def main():
    # Initialize wandb
    wandb.login()
    run = wandb.init(project="dspy")
    
    gc.collect()
    torch.cuda.empty_cache()

    with open("config.json", "r") as f:
        config = json.load(f)["wood"]
    gt_dataset = load_dataset(
        config["GT_IMAGES_DIRECTORY_PATH"],
        config["GT_ANNOTATIONS_DIRECTORY_PATH"],
        config["GT_DATA_YAML_PATH"],
    )

    # Initial annotation with Florence2
    initial_prompt = "Analyze the visual data of a blue stain on a surface, focusing on identifying the presence of cracks, dead knots, missing knots, and a knot with cracks."
    gt_class, TP, FP, FN, acc, F1, dataset = label_images(config, gt_dataset, initial_prompt)
    log_wandb_metrics({
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Accuracy": acc,
        "F1 Score": F1,
    })
    
    # Review annotations directly with Qwen
    ds_review = dataset
    gc.collect()
    torch.cuda.empty_cache()
    # Initialize Qwen model directly with transformers
    print("Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",  # Using 7B instead of 72B for resource compatibility
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )
    print("Qwen2.5-VL model loaded successfully.")
    for image_path, image, annotation in ds_review:
        image = Image.open(image_path)
        boxes = annotation.xyxy
        print("Processing image:", image_path)
        for i, bbox in enumerate(boxes):
            # Convert bbox to a tuple of integers
            box_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            # Crop image based on the current bbox
            cropped_image = image.crop(box_tuple)
            object_name = "defect or anomaly"  # Replace with the actual object name if available
            
            # Direct review with Qwen
            is_valid = review_detection_with_qwen(
                model,
                processor, 
                image, 
                box_tuple, 
                cropped_image,
                object_name
            )
            
            print(f"Qwen review result: {is_valid}")
            
            # Adjust detection confidence based on reviewer decision
            annotation.confidence[i] = 1 if is_valid else 0

    # Filter ds_review based on the updated confidence values
    filtered_ds = filter_detection_dataset(ds_review, threshold=0.5)
    confusion_matrix, acc, map_result = evaluate_detections(filtered_ds, gt_dataset)
    TP = confusion_matrix[0, 0] / confusion_matrix.sum()
    FP = confusion_matrix[0, 1] / confusion_matrix.sum()
    FN = confusion_matrix[1, 0] / confusion_matrix.sum()
    F1 = 2 * TP / (2 * TP + FP + FN)

    log_wandb_metrics({
        "Filtered Accuracy": acc[0],
        "Filtered F1 Score": F1,
        "Filtered TP": TP,
        "Filtered FP": FP,
        "Filtered FN": FN,
    })
    log_wandb_metrics({
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Accuracy": acc,
        "F1 Score": F1,
    })
    print(f"Accuracy after filtering: {acc}")
    print(f"Confusion Matrix after filtering: {confusion_matrix}")


if __name__ == "__main__":
    main()
