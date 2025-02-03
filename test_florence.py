import torch
import json

import os
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)
from utils.Florence_train import Florence2Trainer, Florence2
import wandb
from utils.check_labels import *
from autodistill.detection import CaptionOntology
with open('/zhome/4a/b/137804/Desktop/autolbl/config.json', 'r') as f:
    config = json.load(f)["wood"]

wandb.login()
wandb.init(project="finetune", name=f"{"Florence"}_{"fine_tune"}", tags=["fine_tune", "florence", "bottle"])

#clear the cache
torch.cuda.empty_cache()
model = Florence2Trainer()
wandb.watch(model.model, log_freq=1)
dataset_path = f"{os.getcwd()}/data/wood copy/florence_annotations"

model.train(dataset_path=dataset_path, epochs=2)
ont_list = {
    "color": "color",
    "combined": "combined",
    "hole":"hole",
    "liquid":"liquid",
    "scratch":"liquid"
}
torch.cuda.empty_cache()

base_model = Florence2(ontology=CaptionOntology(ont_list))
#load the model from the checkpoint
base_model.load_checkpoint(config['CHECKPOINT_PATH'])
#label the validation set
dataset_path = f"{os.getcwd()}/data/wood copy/florence_annotations/valid"


dataset = base_model.label(
    input_folder=config['IMAGE_DIR_PATH'],
    extension=".jpg",
    output_folder=config['DATASET_DIR_PATH'])
#check if the dataset is empty
if len(dataset) == 0:

    dataset = base_model.label(
        input_folder=config['IMAGE_DIR_PATH'],
        extension=".png",
        output_folder=config['DATASET_DIR_PATH']
    )

print("Model loaded")  