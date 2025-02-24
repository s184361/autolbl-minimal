import os
import json
import argparse
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image
import requests
import subprocess
import pandas as pd
import wandb
import dspy
import fal_client
import supervision as sv
from utils.check_labels import *
from run_any2 import run_any_args
from llama_cpp import Llama
from utils.wandb_utils import *
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

def metric_function(prompt: str):
    """
    Metric function to evaluate the output of the program.
    Returns the accuracy value.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)["defects"]
    gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])
    gt_class, TP, FP, FN, acc, F1 = label_images(config=config, gt_dataset=gt_dataset, prompt=prompt)
    return acc

def label_images(config: None, gt_dataset: sv.DetectionDataset, prompt: str):
    args = argparse.Namespace(
        config='/zhome/4a/b/137804/Desktop/autolbl/config.json',
        section='defects',
        model='Florence',
        tag='default',
        sahi=False,
        reload=False,
        ontology=f'{prompt}: defect',
        wandb=False,
    )
    run_any_args(args)
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=config['IMAGES_DIRECTORY_PATH'],
        ontology=f'{prompt}: defect',
        wandb=False,
    )
    run_any_args(args)
    dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=config['IMAGES_DIRECTORY_PATH'],
            annotations_directory_path=config['ANNOTATIONS_DIRECTORY_PATH'],
            data_yaml_path=config['DATA_YAML_PATH']
        )
    
    confusion_matrix, acc, map_result=evaluate_detections(dataset, gt_dataset)
    #compare_plot(dataset, gt_dataset)
    #extract true positives
    print(f"Accuracy: {acc}")
    # return "class", "TP", "FP", "FN", "Accuracy", "F1"
    gt_class = "defect"
    TP = confusion_matrix[0, 0]/confusion_matrix.sum()
    FP = confusion_matrix[0, 1]/confusion_matrix.sum()
    FN = confusion_matrix[1, 0]/confusion_matrix.sum()
    F1 = 2*TP/(2*TP+FP+FN)

    return gt_class, TP, FP, FN, acc[0], F1 

class CheckAndRevisePrompt(dspy.Signature):
    """Signature for checking and revising prompts"""
    desired_score = dspy.InputField()
    current_score = dspy.InputField()
    current_prompt = dspy.InputField()
    best_prompt = dspy.InputField()
    best_score = dspy.InputField()
    context = dspy.InputField()
    
    revised_prompt = dspy.OutputField()
    feedback = dspy.OutputField()

def main():

    # Initialize wandb
    wandb.login()
    run = wandb.init(project="dspy")
    load_dotenv()

    # Optional
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
    os.environ["ANTHROPIC_API_KEY"] = (
        "os.getenv("ANTHROPIC_API_KEY", "")"
    )

    #llm = Llama(model_path="/work3/s184361/model/zephyr-7b-beta.Q4_0.gguf",n_gpu_layers=-1,n_ctx=0,verbose=False)
    #lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434")
    lm = dspy.LM('anthropic/claude-3-opus-20240229')
    dspy.configure(lm=lm)
    #llamalm = LlamaCpp(model="llama", llama_model=llm,  model_type="chat", temperature=0.4)
    #dspy.settings.configure(lm=llamalm)

    # Assume you have your own train and validation sets
    trainset = [
        {"input" : "defect", "output": 0.2}
    ]

    devset = [
        {"input" : "defect", "output": 0.2},
        # Add more examples
    ]

    # Define a task module
    class MyTask(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("input -> output")

        def forward(self, input):
            return self.prog(input=input)

    # Evaluate the baseline program
    evaluate = Evaluate(devset=devset[:], metric=metric_function, num_threads=8, display_progress=True, display_table=False)
    program = MyTask()
    evaluate(program, devset=devset[:])

    # Optimizing with MIPROv2
    teleprompter = MIPROv2(
        metric=metric_function,
        auto="light",  # Can choose between light, medium, and heavy optimization runs
    )

    # Optimize program
    print(f"Optimizing program with MIPRO...")
    optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        requires_permission_to_run=False,
    )

    # Save optimized program for future use
    optimized_program.save(f"mipro_optimized")

    # Evaluate optimized program
    print(f"Evaluate optimized program...")
    evaluate(optimized_program, devset=devset[:])
    
if __name__ == "__main__":
    main()