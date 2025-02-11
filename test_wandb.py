import json
import supervision as sv
from utils.check_labels import *
from run_any2 import run_any_args
from llama_cpp import Llama
import pandas as pd
from utils.wandb_utils import *
import wandb
import argparse
import subprocess

    # Load the dataset
def main():
    # Initialize wandb
    wandb.login()
    run = wandb.init(project="debug_wandb")
    with open('config.json', 'r') as f:
        config = json.load(f)["defects"]
    gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=config['IMAGES_DIRECTORY_PATH'],
            annotations_directory_path=config['ANNOTATIONS_DIRECTORY_PATH'],
            data_yaml_path=config['DATA_YAML_PATH']
        )
    wandb_tab = compare_plot(dataset, gt_dataset, run)
    run.log({"comparison_images": wandb_tab})
    run.finish()

if __name__ == "__main__":
    main()