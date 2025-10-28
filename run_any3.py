import argparse
import json
import os
import re
import torch
#import cv2
import supervision as sv
#import matplotlib.pyplot as plt
from utils.check_labels import *
from autodistill.detection import CaptionOntology
# from autodistill_grounding_dino import GroundingDINO
from utils.Florence_fixed import Florence2

from utils.composed_detection_model import ComposedDetectionModel2
#from utils.detection_base_model import DetectionBaseModel
from utils.embedding_ontology import EmbeddingOntologyImage
from utils.wandb_utils import compare_plot as compare_wandb
import wandb
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run autodistill with specified configuration.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the JSON configuration file.')
    parser.add_argument('--section', type=str, default='defects', help='Section of the configuration to use.')
    parser.add_argument('--model', type=str, choices=['DINO', 'Florence', 'SAMHQ', 'Combined', 'MetaCLIP', "Qwen"], default='Florence', help='Model to use for autodistill.')
    parser.add_argument('--tag', type=str, default='default', help='Tag for the wandb run.')
    parser.add_argument('--sahi', action='store_true', help='Use SAHI for inference.')
    parser.add_argument('--reload', action='store_true', help='Reload the dataset from YOLO format.')
    parser.add_argument('--ontology', type=str, default='', help='Path to the ontology file.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--save_images', action='store_true', help='Save images for destillation')
    parser.add_argument('--nms', type=str, default="no_nms", help='NMS setting for the model.')
    parser.add_argument('--group', type=str, default=None, help='Group for the wandb run.')
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

    if args.wandb:
        tags = []
        for key, value in vars(args).items():
            # Skip ontology in tags since it's stored in config
            if key == 'ontology':
                continue
            # Truncate other long values
            if isinstance(value, str) and len(f"{key}: {value}") > 63:
                tag = f"{key}: {value[:40]}..."
            else:
                tag = f"{key}: {value}"
            tags.append(tag)
        wandb.login()
        wandb.init(project="Thesis", name=f"{args.model}_{args.tag}", tags=tags, config=config,group=args.group)
        #update the config with the args
        wandb.config.update(args)
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

    if args.ontology in ["", None]:
        # Define ontology

        with open("data/Semantic Map Specification.txt", "r") as file:
            content = file.read()
        names = re.findall(r"name=([^\n]+)", content)
        names = sorted([name.lower().replace("_", " ") for name in names])
        ont_list = {name: name for name in names}
        #ont_list = {"defect": "defect"}
        print(ont_list)
    elif args.ontology == "BAG_OF_WORDS":
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
        "wood surface blistering": "blistering",
        "cracked glass": "cracked glass",
        "broken glass": "broken glass",
        "glass scratch": "glass scratch",
        "glass chip": "glass chip",
        "glass stain": "glass stain",
        "broken mirror": "broken mirror",
        "bottle neck": "bottle neck",
        "glass shard": "glass shard",
        "glass edge": "glass edge",
        "anomalous glass": "anomalous glass",
        "defective glass": "defective glass",
    }
    else:
        try:
            ont_list = dict(item.split(": ") for item in args.ontology.split(", "))

        except:
            ont_list = {args.ontology: "defect"}
        print(args.ontology)
        print(ont_list)
    # Initialize the model
    if args.model == "DINO":
        from utils.grounding_dino_model import GroundingDINO
        base_model = GroundingDINO(ontology=CaptionOntology(ont_list))
    elif args.model == "Florence":
        if args.wandb:
            from autodistill_florence_2 import Florence2
        base_model = Florence2(ontology=CaptionOntology(ont_list))
    elif args.model == "SAMHQ":
        from autodistill_sam_hq.samhq_model import SAMHQ
        base_model = SAMHQ(CaptionOntology(ontology=ont_list))
    elif args.model == "Combined":
        from utils.grounding_dino_model import GroundingDINO
        from utils.metaclip_model import MetaCLIP
        detection_model = GroundingDINO(ontology=CaptionOntology({config["PROMPT"]: "defect"}))
        classification_model = MetaCLIP(ontology=CaptionOntology(ont_list))
        base_model = ComposedDetectionModel2(detection_model=detection_model, classification_model=classification_model)
    elif args.model == "Combined2":
        from utils.metaclip_model import MetaCLIP
        detection_model = Florence2(ontology=CaptionOntology({config["PROMPT"]: "defect"}))
        classification_model = MetaCLIP(ontology=CaptionOntology(ont_list))
        base_model = ComposedDetectionModel2(detection_model=detection_model, classification_model=classification_model)
    elif args.model == "MetaCLIP":
        from utils.metaclip_model import MetaCLIP
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
    elif loaded_model is not None:
        ("Using loaded model")
        base_model = loaded_model
        #update the ontology
        base_model.ontology = CaptionOntology(ont_list) 
    elif args.model == "Qwen":
        from utils.qwen25_model import Qwen25VL
        print("Load Qwen model")
        base_model = Qwen25VL(ontology=CaptionOntology(ont_list),hf_token="os.getenv("HF_TOKEN", "")")

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
        #wandb.log({"Prompt Table": table})
    except:
        print("No wandb")

    dataset = base_model.label(
        input_folder=config['IMAGE_DIR_PATH'],
        extension=".jpg",
        output_folder=config['DATASET_DIR_PATH'],
        sahi=args.sahi,
        save_images=args.save_images,
        nms_settings=args.nms
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


    # Finish the wandb run
    if args.wandb:
        gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
        #set one class for the gt_dataset if ont_list is {args.ontology: "defect"}
        if len(ont_list) == 1 and list(ont_list.values())[0] == "defect" or args.ontology == "BAG_OF_WORDS":
            gt_dataset = set_one_class(gt_dataset)
            #check if the gt_dataset is correct
            print("Dataset correct:", check_classes(gt_dataset))
        confusion_matrix, precision, recall, F1, map05,map05095 =evaluate_detections(dataset, gt_dataset)
        print(f"Confusion matrix: {confusion_matrix}")
        
        print(f"recall: {recall}")
        if len(recall) > 2:
            # Take the last accuracy value
            recall = recall[-1]
            precision = precision[-1]
            F1 = F1[-1]
        else:
            # If there is only one accuracy value, use it directly
            recall = recall[0]
            precision = precision[0]
            F1 = F1[0]
        if len(dataset)<100:
            compare_wandb(dataset, gt_dataset)
        wandb.log({                
            "recall": recall,
            "precision": precision,
            "F1": F1,
            "confusion_matrix": confusion_matrix,
            })
    return dataset

def main():
    args = parse_arguments()
    #set section to work3_tires
    #args.section = "wood"
    #args.ontology = "pumps tensed oceanuses [unused810] [unused368] bombay wavelengthsctuseriantlanurianbant yells"

    run_any_args(args)
if __name__ == "__main__":
    main() 