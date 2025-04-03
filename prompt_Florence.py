import argparse
import json
import os
import re
import torch
import cv2
import supervision as sv
import matplotlib.pyplot as plt
from utils.check_labels import *

# from autodistill_grounding_dino import GroundingDINO
from utils.Florence_fixed import Florence2Prompt, Florence2
from autodistill.detection import CaptionOntology

from utils.wandb_utils import compare_plot as compare_wandb
import wandb
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run autodistill with specified configuration.")
    parser.add_argument('--config', type=str, default='/zhome/4a/b/137804/Desktop/autolbl/config.json', help='Path to the JSON configuration file.')
    parser.add_argument('--section', type=str, default='defects', help='Section of the configuration to use.')
    parser.add_argument('--model', type=str, choices=['DINO', 'Florence', 'SAMHQ', 'Combined', 'MetaCLIP', "Qwen"], default='Florence', help='Model to use for autodistill.')
    parser.add_argument('--tag', type=str, default='default', help='Tag for the wandb run.')
    parser.add_argument('--sahi', action='store_true', help='Use SAHI for inference.')
    parser.add_argument('--reload', type=bool, default=False, help='Reload the dataset from YOLO format.')
    parser.add_argument('--ontology', type=str, default='', help='Path to the ontology file.')
    parser.add_argument('--wandb', type=bool, default=True, help='Use wandb for logging')
    parser.add_argument('--save_images', type=bool, default=False, help='Save images for destillation')
    return parser.parse_args()
def set_one_class(gt_dataset):
    for key in gt_dataset.annotations.keys():
        gt_dataset.annotations[key].class_id = np.zeros_like(gt_dataset.annotations[key].class_id)
    gt_dataset.classes = ['defect']
    return gt_dataset

def check_classes(gt_dataset):
    for key in gt_dataset.annotations.keys():
        for i in range(len(gt_dataset.annotations[key])):
            if gt_dataset.annotations[key][i].class_id != len(gt_dataset.classes) - 1:
                return False
    return True
def run_any_args(args,loaded_model=None):
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)[args.section]

    # Initialize wandb
    if args.wandb:
        wandb.login()
        wandb.init(project="Florence_prompt_backprop", name=f"{args.model}_{args.tag}", tags=[args.tag])

    # Reset folders
    try:
        print("Resetting folders")
        #reset_folders(config['DATASET_DIR_PATH'], config.get('RESULTS_DIR_PATH', 'results'))
    except:
        print("No folders to delete")

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Display image sample
    image_paths = sv.list_files_with_extensions(
        directory=config['IMAGE_DIR_PATH'],
        extensions=["bmp", "jpg", "jpeg", "png"]
    )
    print('Image count:', len(image_paths))

    # titles = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths[:config['SAMPLE_SIZE']]]
    # images = [cv2.imread(image_path) for image_path in image_paths[:config['SAMPLE_SIZE']]]
    # plt.ion()
    # sv.plot_images_grid(images=images, titles=titles, grid_size=config['SAMPLE_GRID_SIZE'], size=config['SAMPLE_PLOT_SIZE'])
    # plt.savefig(os.path.join(config.get('RESULTS_DIR_PATH', 'results'), "sample_images_grid.png"))

    if args.ontology in ["", None]:
        # Define ontology

        with open("data/Semantic Map Specification.txt", "r") as file:
            content = file.read()
        names = re.findall(r"name=([^\n]+)", content)
        names = sorted([name.lower().replace("_", " ") for name in names])
        ont_list = {name: name for name in names}

    else:
        try:
            ont_list = dict(item.split(": ") for item in args.ontology.split(", "))

        except:
            ont_list = {args.ontology: "defect"}
        print(args.ontology)
        print(ont_list)
    # Initialize the model
    base_model = Florence2Prompt(initial_prompt=args.ontology)
    # Log model settings
    try:
        wandb.config.update({
            "model": args.model,
            "ontology": ont_list,
            "input_folder": config['IMAGE_DIR_PATH'],
            "output_folder": config['DATASET_DIR_PATH'],
            "sahi": args.sahi
        })
        table = wandb.Table(columns=["prompt", "caption"])
        for key, value in ont_list.items():
            table.add_data(key, value)
    except:
        print("No wandb")

    # Load the dataset
    gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    final_prompt = base_model.train(ds_train=gt_dataset, ds_valid=gt_dataset, epochs=350)
    #label the dataset
    base_model = Florence2(ontology=CaptionOntology({final_prompt: "defect"})) 
    dataset = base_model.label(
        input_folder=config['IMAGE_DIR_PATH'],
        extension=".jpg",
        output_folder=config['DATASET_DIR_PATH'],
        sahi=args.sahi,
        save_images=args.save_images
    )
        # check if the dataset is empty
    if len(dataset) == 0:

        dataset = base_model.label(
            input_folder=config['IMAGE_DIR_PATH'],
            extension=".png",
            output_folder=config['DATASET_DIR_PATH'],
            sahi=args.sahi,
            save_images=args.save_images
        )
    print("Final prompt:", final_prompt)
    if args.reload:
        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=config['IMAGES_DIRECTORY_PATH'],
            annotations_directory_path=config['ANNOTATIONS_DIRECTORY_PATH'],
            data_yaml_path=config['DATA_YAML_PATH']
        )

    print("Dataset size:", len(dataset))
    # Log the size of the dataset
    try:
        wandb.log({"dataset_size": len(dataset)})
    except:
        pass
    # Plot annotated images

    # Evaluate the dataset
    print(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    if len(dataset)<100:
        # 
        pass

    # Finish the wandb run
    if args.wandb:
        gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
        #set one class for the gt_dataset
        gt_dataset = set_one_class(gt_dataset)
        #check if the gt_dataset is correct
        print("Dataset correct:", check_classes(gt_dataset))
        confusion_matrix, acc, map_result=evaluate_detections(dataset, gt_dataset)
        print(f"Confusion matrix: {confusion_matrix}")
        acc = acc[0]
        print(f"Accuracy: {acc}")
        gt_class = "defect"
        TP = confusion_matrix[0, 0] / confusion_matrix.sum()
        FP = confusion_matrix[0, 1] / confusion_matrix.sum()
        FN = confusion_matrix[1, 0] / confusion_matrix.sum()
        F1 = 2 * TP / (2 * TP + FP + FN)
        compare_wandb(dataset, gt_dataset)
        wandb.log({                
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "accuracy": acc,
                "F1": F1
            })
    return dataset

def main():
    args = parse_arguments()
    #set section to work3_tires
    args.section = "wood"
    args.ontology = "<s>remainingsounding・�.,''brainer μ j glim️ Kurdistan Scalia</s>"

    run_any_args(args)
if __name__ == "__main__":
    main() 
