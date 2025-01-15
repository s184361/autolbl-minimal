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
    parser.add_argument('--config', type=str, default='/zhome/4a/b/137804/Desktop/autolbl/config.json', help='Path to the JSON configuration file.')
    parser.add_argument('--section', type=str, default='defect', help='Section of the configuration to use.')
    parser.add_argument('--model', type=str, choices=['DINO', 'Florence', 'SAMHQ', 'Combined', 'MetaCLIP'], default='DINO', help='Model to use for autodistill.')
    parser.add_argument('--tag', type=str, default='default', help='Tag for the wandb run.')
    parser.add_argument('--sahi', action='store_true', help='Use SAHI for inference.')
    parser.add_argument('--reload', action='store_true', help='Reload the dataset from YOLO format.')
    parser.add_argument('--ontology', type=str, default='', help='Path to the ontology file.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)[args.section]

    # Initialize wandb
    wandb.login()
    wandb.init(project="auto_new_wood_annotations", name=f"{args.model}_{args.tag}", tags=[args.tag])

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

    if args.ontology in ["", None]:
        # Define ontology
        
        with open("data/Semantic Map Specification.txt", "r") as file:
            content = file.read()
        names = re.findall(r"name=([^\n]+)", content)
        names = sorted([name.lower().replace("_", " ") for name in names])
        ont_list = {name: name for name in names}
        print(ont_list)
        """
        ont_list =ont_list = {
        "wood defect": "defect",
        "wood grain": "grain",
        "cat": "cat",
        "dog": "dog",
        "cracked wood": "crack",
        "blurred wood grain": "blurred grain",
        "contamination on wood": "contamination",
        "anomaly in wood": "anomaly",
        "spec on wood": "spec",
        "smooth wood": "smooth",
        "damaged wood surface": "damaged surface",
        "sharp wood splinter": "sharp splinter",
        "missing wood piece": "missing piece",
        "dirty wood surface": "dirty surface",
        "chip in wood": "chip",
        "wood discoloration": "discoloration",
        "foreign particle on wood": "particle",
        "residue on wood": "residue",
        "broken wood section": "broken section",
        "polished wood": "polished",
        "perfect wood grain": "perfect grain",
        "rough wood texture": "rough texture",
        "micro-cracks in wood": "micro-cracks",
        "a knot in wood": "knot",
        "wood grain pattern": "grain pattern",
        "splintered wood": "splinter",
        "blurry wood defect": "blurry defect",
        "sharp wood edge": "sharp edge",
        "rough wood edge": "rough edge",
        "soft wood section": "soft wood",
        "hardwood": "hardwood",
        "a saw mark": "saw mark",
        "wood scratch": "scratch",
        "wood dent": "dent",
        "fuzzy wood grain": "fuzzy grain",
        "clean wood surface": "clean surface",
        "wood rot": "rot",
        "fungus on wood": "fungus",
        "mold on wood": "mold",
        "wood chip": "chip",
        "rough cut": "rough cut",
        "smooth cut": "smooth cut",
        "grain misalignment": "misaligned grain",
        "split wood": "split",
        "blurry wood imperfection": "blurry imperfection",
        "sharp wood imperfection": "sharp imperfection",
        "uneven wood surface": "uneven surface",
        "peeling wood finish": "peeling finish",
        "cracked wood surface": "cracked surface",
        "warped wood": "warped",
        "bowed wood": "bowed",
        "cupped wood": "cupped",
        "twisted wood": "twisted",
        "burn mark on wood": "burn mark",
        "water stain on wood": "water stain",
        "wood scratch marks": "scratch marks",
        "gouge in wood": "gouge",
        "faded wood color": "faded color",
        "blurry knot": "blurry knot",
        "sharp knot": "sharp knot",
        "small crack in wood": "small crack",
        "large crack in wood": "large crack",
        "wood filler residue": "filler residue",
        "loose grain": "loose grain",
        "tight grain": "tight grain",
        "natural wood imperfection": "natural imperfection",
        "blurry grain boundary": "blurry boundary",
        "sharp grain boundary": "sharp boundary",
        "chipped wood corner": "chipped corner",
        "frayed wood edge": "frayed edge",
        "smooth wood finish": "smooth finish",
        "rough wood finish": "rough finish",
        "blurry saw mark": "blurry saw mark",
        "clear saw mark": "clear saw mark",
        "wood decay": "decay",
        "insect damage in wood": "insect damage",
        "termite holes in wood": "termite holes",
        "sharp splinters": "sharp splinters",
        "blurry discoloration": "blurry discoloration",
        "sharp discoloration": "sharp discoloration",
        "sun-bleached wood": "sun-bleached",
        "varnish peeling": "peeling varnish",
        "paint residue on wood": "paint residue",
        "wood putty mark": "putty mark",
        "wood patch": "patch",
        "wood veneer": "veneer",
        "a plank of wood": "plank",
        "a tree trunk": "tree trunk",
        "a wooden board": "board",
        "a wooden beam": "beam",
        "a wooden table": "table",
        "a wooden chair": "chair",
        "fire damage on wood": "fire damage",
        "charring on wood": "charring",
        "wood grain variation": "grain variation",
        "uneven grain": "uneven grain",
        "fine cracks in wood": "fine cracks",
        "deep cracks in wood": "deep cracks",
        "resin pocket in wood": "resin pocket",
        "wood glue residue": "glue residue",
        "wood surface blistering": "blistering"
    }
    """
    else:
        print(args.ontology)
        ont_list = dict(item.split(": ") for item in args.ontology.split(", "))
        print(ont_list)
    # Initialize the model
    if args.model == "DINO":
        base_model = GroundingDINO(ontology=CaptionOntology(ont_list))
    elif args.model == "Florence":
        base_model = Florence2(ontology=CaptionOntology(ont_list))
    elif args.model == "SAMHQ":
        base_model = SAMHQ(CaptionOntology(ontology=ont_list))
    elif args.model == "Combined":
        detection_model = GroundingDINO(ontology=CaptionOntology({config["PROMPT"]: "defect"}))
        classification_model = MetaCLIP(ontology=CaptionOntology(ont_list))
        base_model = ComposedDetectionModel2(detection_model=detection_model, classification_model=classification_model)
    elif args.model == "Combined2":
        detection_model = Florence2(ontology=CaptionOntology({config["PROMPT"]: "defect"}))
        classification_model = MetaCLIP(ontology=CaptionOntology(ont_list))
        base_model = ComposedDetectionModel2(detection_model=detection_model, classification_model=classification_model)
    elif args.model == "MetaCLIP":
        HOME2 = os.getcwd()
        images_to_classes = {
            os.path.join(f"{HOME2}/croped_images", "100000010_live knot.jpg"): "knot",
            os.path.join(f"{HOME2}/croped_images", "100000009_dead knot.jpg"): "knot",
            os.path.join(f"{HOME2}/croped_images", "knot missing.jpg"): "knot missing",
            os.path.join(f"{HOME2}/croped_images", "100000034_knot with crack.jpg"): "knot with crack",
            os.path.join(f"{HOME2}/croped_images", "100000074_crack.jpg"): "crack",
            os.path.join(f"{HOME2}/croped_images", "100000000_quartzity.jpg"): "quartzity",
            os.path.join(f"{HOME2}/croped_images", "100000013_resin.jpg"): "resin",
            os.path.join(f"{HOME2}/croped_images", "100000002_marrow.jpg"): "marrow",
            os.path.join(f"{HOME2}/croped_images", "overgrown.jpg"): "overgrown",
            os.path.join(f"{HOME2}/croped_images", "blue stain.jpg"): "blue stain"
        }
    # Create embedding ontology and models
        images_to_classes = dict(sorted(images_to_classes.items(), key=lambda item: item[1]))
        img_emb = EmbeddingOntologyImage(images_to_classes)
        base_model = MetaCLIP(img_emb)
        

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
        # Log the table
        wandb.log({"Prompt Table": table})
    except:
        print("No wandb")

    dataset = base_model.label(
        input_folder=config['IMAGE_DIR_PATH'],
        extension=".jpg",
        output_folder=config['DATASET_DIR_PATH'],
        sahi=args.sahi
    )
    #check if the dataset is empty
    if len(dataset) == 0:

        dataset = base_model.label(
            input_folder=config['IMAGE_DIR_PATH'],
            extension=".png",
            output_folder=config['DATASET_DIR_PATH'],
            sahi=args.sahi
        )
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
    plot_annotated_images(dataset, config['SAMPLE_SIZE'], os.path.join(config.get('RESULTS_DIR_PATH', 'results'), "sample_annotated_images_grid.png"))

    # Evaluate the dataset
    #update_labels(config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    print(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    print("GT Dataset size:", len(gt_dataset))
    compare_classes(gt_dataset, dataset)
    #compare_image_keys(gt_dataset, dataset)
    evaluate_detections(dataset, gt_dataset)
    if len(dataset)<100:
        compare_plot(dataset, gt_dataset)



    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 