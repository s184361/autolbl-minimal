import dspy
import os
from PIL import Image
from io import BytesIO
import requests
import fal_client
from dotenv import load_dotenv
import subprocess
import json
import supervision as sv
from utils.check_labels import *
from run_any2 import run_any_args
import argparse
def label_images(config: None, gt_dataset: sv.DetectionDataset, prompt: str):
    #python run_any2.py --section defects --model Florence --ontology
    # Create the arguments
    args = argparse.Namespace(
        config='/zhome/4a/b/137804/Desktop/autolbl/config.json',
        section='defects',
        model='Florence',
        tag='default',
        sahi=False,
        reload=False,
        ontology=f'{prompt}: defect'
    )
    run_any_args(args)
    dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=config['IMAGES_DIRECTORY_PATH'],
            annotations_directory_path=config['ANNOTATIONS_DIRECTORY_PATH'],
            data_yaml_path=config['DATA_YAML_PATH']
        )
    
    confusion_matrix, acc, map_result=evaluate_detections(dataset, gt_dataset)
    #extract true positives
    print(f"Accuracy: {acc}")
    TP = confusion_matrix[0, 0]/confusion_matrix.sum()
    return  TP

load_dotenv()

# Optional
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["ANTHROPIC_API_KEY"] = (
    "os.getenv("ANTHROPIC_API_KEY", "")"
)

#lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434")
lm = dspy.LM('anthropic/claude-3-opus-20240229')

dspy.configure(lm=lm)

class CheckAndRevisePrompt(dspy.Signature):
    """Signature for checking and revising prompts"""
    desired_score = dspy.InputField()
    current_score = dspy.InputField()
    current_prompt = dspy.InputField()
    context = dspy.InputField()
    
    revised_prompt = dspy.OutputField()
    feedback = dspy.OutputField()

#autolbl
with open('config.json', 'r') as f:
    config = json.load(f)["defects"]
gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])

check_and_revise_prompt = dspy.Predict(CheckAndRevisePrompt)

initial_prompt = "defect anomaly scratch crack split knot dead knot in wood"
current_prompt = initial_prompt

max_iter = 5
for i in range(max_iter):
    print(f"Iteration {i+1} of {max_iter}")
    current_score = label_images(config=config, gt_dataset=gt_dataset, prompt = current_prompt)
    result = check_and_revise_prompt(
        desired_score=1.0,
        current_score=current_score,
        current_prompt=current_prompt,
        context = "You are designing a prompt to help a vison model identify defects in wood images"
    )
    print(f"Current score: {current_score}")
    if current_score == 1:
        break
    else:
        current_prompt = result.revised_prompt
        #remove , from prompt
        current_prompt = current_prompt.replace(",", "")
        print(f"Feedback: {result.feedback}")
        print(f"Revised prompt: {result.revised_prompt}")

print(f"Final prompt: {current_prompt}")