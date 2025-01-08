import argparse
import json
import os
import re
import torch
import cv2
import supervision as sv
import matplotlib.pyplot as plt
from utils.check_labels import *
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from autodistill_florence_2 import Florence2
from autodistill_sam_hq.samhq_model import SAMHQ
from utils.composed_detection_model import ComposedDetectionModel2
from utils.embedding_ontology import EmbeddingOntologyImage
from utils.metaclip_model import MetaCLIP
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run autodistill with specified configuration.")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('--section', type=str, required=True, help='Section of the configuration to use.')
    parser.add_argument('--model', type=str, choices=['DINO', 'Florence', 'SAMHQ', 'Combined', 'MetaCLIP'], required=True, help='Model to use for autodistill.')
    parser.add_argument('--tag', type=str, default='default', help='Tag for the wandb run.')
    parser.add_argument('--sahi', action='store_true', help='Use SAHI for inference.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)[args.section]

    # Initialize wandb
    wandb.login()
    wandb.init(project="auto_label", name=f"{args.model}_{args.tag}", tags=[args.tag])

    # Reset folders
    reset_folders(config['DATASET_DIR_PATH'], config.get('RESULTS_DIR_PATH', 'results'))

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Display image sample
    image_paths = sv.list_files_with_extensions(
        directory=config['IMAGE_DIR_PATH'],
        extensions=["bmp", "jpg", "jpeg", "png"]
    )
    print('Image count:', len(image_paths))

    titles = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths[:config['SAMPLE_SIZE']]]
    images = [cv2.imread(image_path) for image_path in image_paths[:config['SAMPLE_SIZE']]]
    plt.ion()
    sv.plot_images_grid(images=images, titles=titles, grid_size=config['SAMPLE_GRID_SIZE'], size=config['SAMPLE_PLOT_SIZE'])
    plt.savefig(os.path.join(config.get('RESULTS_DIR_PATH', 'results'), "sample_images_grid.png"))

    # Define ontology
    with open("data/Semantic Map Specification.txt", "r") as file:
        content = file.read()
    names = re.findall(r"name=([^\n]+)", content)
    names = [name.lower().replace("_", " ") for name in names]
    ont_list = {name: name for name in names}
    print(ont_list)

    # Initialize the model
    if args.model == "DINO":
        base_model = GroundingDINO(ontology=CaptionOntology(ont_list))
    elif args.model == "Florence":
        base_model = Florence2(ontology=CaptionOntology(ont_list))
    elif args.model == "SAMHQ":
        base_model = SAMHQ(CaptionOntology(ontology=ont_list))
    elif args.model == "Combined":
        detection_model = GroundingDINO(ontology=CaptionOntology(ont_list))
        classification_model = MetaCLIP(EmbeddingOntologyImage(ont_list))
        base_model = ComposedDetectionModel2(detection_model=detection_model, classification_model=classification_model)
    elif args.model == "MetaCLIP":
        base_model = MetaCLIP(EmbeddingOntologyImage(ont_list))

    # Log model settings
    wandb.config.update({
        "model": args.model,
        "ontology": ont_list,
        "input_folder": config['IMAGE_DIR_PATH'],
        "output_folder": config['DATASET_DIR_PATH'],
        "sahi": args.sahi
    })

    # Label the dataset
    dataset = base_model.label(
        input_folder=config['IMAGE_DIR_PATH'],
        extension=".png",
        output_folder=config['DATASET_DIR_PATH'],
        sahi=args.sahi
    )

    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=config['IMAGES_DIRECTORY_PATH'],
        annotations_directory_path=config['ANNOTATIONS_DIRECTORY_PATH'],
        data_yaml_path=config['DATA_YAML_PATH']
    )
    print("Dataset size:", len(dataset))

    # Plot annotated images
    plot_annotated_images(dataset, config['SAMPLE_SIZE'], os.path.join(config.get('RESULTS_DIR_PATH', 'results'), "sample_annotated_images_grid.png"))

    # Evaluate the dataset
    update_labels(config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    compare_classes(gt_dataset, dataset)
    compare_image_keys(gt_dataset, dataset)
    evaluate_detections(dataset, gt_dataset)
    compare_plot(dataset, gt_dataset)

    # Log the size of the dataset
    wandb.log({"dataset_size": len(dataset)})

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 