import dspy
import os
from PIL import Image
from io import BytesIO

# from dotenv import load_dotenv

import json
import supervision as sv
from utils.check_labels import *
from run_any2 import run_any_args

import argparse
import wandb
from utils.wandb_utils import *
import pandas as pd
import os
import subprocess
import torch
import gc
from utils.check_labels import set_one_class, check_classes

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def label_images(config: None, gt_dataset: sv.DetectionDataset, prompt: str):
    # python run_any2.py --section defects --model Florence --ontology
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
    # ompare_plot(dataset, gt_dataset)
    # extract true positives
    print(f"Accuracy: {acc}")
    # return "class", "TP", "FP", "FN", "Accuracy", "F1"
    gt_class = "defect"
    TP = confusion_matrix[0, 0] / confusion_matrix.sum()
    FP = confusion_matrix[0, 1] / confusion_matrix.sum()
    FN = confusion_matrix[1, 0] / confusion_matrix.sum()
    F1 = 2 * TP / (2 * TP + FP + FN)

    return gt_class, TP, FP, FN, acc[0], F1,dataset

def template_prompt(image_feature, location, region_feature, object_name):
    """Create a text-only prompt since Ferret can't process arrays in text."""
    # Convert coordinates to string representation if needed
    if isinstance(location, tuple):
        location_str = f"at coordinates {location}"
    else:
        location_str = str(location)
        
    prompt = f"Considering the region {location_str} of the image, would you classify it as a {object_name} category? Respond with only 'yes' or 'no'."
    return prompt

class CheckAndRevisePrompt(dspy.Signature):
    """Signature for checking and revising prompts"""

    input_prompt = dspy.InputField()
    output_class = dspy.OutputField(type=bool)
    
class CheckDetection(dspy.Signature):
    """
    Signature for checking and revising object detection prompts.
    This class represents a signature for checking whether a specified region
    within an image contains a certain object with high confidence.
    Attributes:
        input_image (dspy.InputField): The full input image to analyze.
        input_box (dspy.InputField): The bounding box coordinates or description of the region of interest.
        input_region (dspy.InputField): The cropped region of the image to be classified.
        input_object (dspy.InputField): The name of the object category to check for.
        output_class (dspy.OutputField): Boolean output indicating whether the region contains
                                       the specified object without doubt (True) or not (False).
    """

    input_image = dspy.InputField(description="The full input image to analyze.", type=dspy.Image)
    input_box = dspy.InputField(description="The bounding box coordinates or description of the region of interest, in the format (x1, y1, x2, y2).")
    input_region = dspy.InputField(description="The cropped region of the image to be classified. Corresponds to the bounding box coordinates.", type=dspy.Image)
    input_object = dspy.InputField(description="The name of the object category to check for.")
    output_bool = dspy.OutputField(description="Detection correct? True or False", type=bool)

def main():

    # Initialize wandb
    wandb.login()
    run = wandb.init(project="dspy")
    # load_dotenv()

    lm = dspy.LM(
        #"ollama/m/ferret", api_base="http://localhost:11434", api_key="",
        "ollama/qwen2.5:7b", api_base="http://localhost:11434", api_key="",
    )
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
    gt_class, TP, FP, FN, acc, F1,dataset = label_images(config, gt_dataset, initial_prompt)

    ds_review = dataset
    for image_path, _, annotation in ds_review:
        boxes = annotation.xyxy
        print(boxes)
        image = Image.open(image_path)
        key = image_path
        for i, bbox in enumerate(boxes):
            # Convert bbox to tuple of integers
            box_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            
            # Crop image based on the current bbox
            cropped_image = image.crop(box_tuple)
            region_feature = np.array(cropped_image)
            object_name = "defect"
            image_feature = np.array(image)
            prompt = template_prompt(image_feature, box_tuple, region_feature, object_name)

            response = check_and_revise_prompt(input_prompt=prompt)
            response2 = check_detection(
            input_image=dspy.Image.from_PIL(image),
            input_box=box_tuple,
            input_region=dspy.Image.from_PIL(cropped_image),
            input_object=object_name,
            )
            
            print(response)
            print(response2)
            if 'yes' in response:
                #change confidence to 1
                print("ds_review.annotations[key][i].confidence", ds_review.annotations[key][i].confidence)
                print("annotation.confidence[i]", annotation.confidence[i])
                annotation.confidence[i] = 1
                ds_review.annotations[key][i].confidence = 1
            else:
                #change confidence to 0
                print("ds_review.annotations[key][i].confidence", ds_review.annotations[key][i].confidence)
                print("annotation.confidence[i]", annotation.confidence[i])
                annotation.confidence[i] = 0
                ds_review.annotations[key][i].confidence = 0
            print("annotation.confidence[i]", annotation.confidence[i])
    # Now evaluate with the filtered dataset instead of trying to create a single Detections object
    #filter ds_review to only include annotations with confidence > 0.5
    ds_review = ds_review[ds_review.confidence > 0.5]
    confusion_matrix, acc, map_result = evaluate_detections(ds_review, gt_dataset)
    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix: {confusion_matrix}")




if __name__ == "__main__":
    main()
