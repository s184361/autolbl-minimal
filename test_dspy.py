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
from llama_cpp import Llama

import argparse
import wandb
from utils.wandb_utils import *
import pandas as pd
def label_images(config: None, gt_dataset: sv.DetectionDataset, prompt: str):
    #python run_any2.py --section defects --model Florence --ontology
    # Create the arguments
    args = argparse.Namespace(
        config='/zhome/4a/b/137804/Desktop/autolbl/config.json',
        section='defects',
        model='DINO',
        tag='default',
        sahi=False,
        reload=False,
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
    wandb.init(project="dspy")
    load_dotenv()

    # Optional
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
    os.environ["ANTHROPIC_API_KEY"] = (
        "os.getenv("ANTHROPIC_API_KEY", "")"
    )

    llm = Llama(
        model_path="./sppo_finetuned_llama_3_8b.gguf",
        n_gpu_layers=-1,
        n_ctx=0,
        verbose=False
    )
    #lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434")
    #lm = dspy.LM('anthropic/claude-3-opus-20240229')
    #dspy.configure(lm=lm)
    llamalm = dspy.LlamaCpp(model="llama", llama_model=llm,  model_type="chat", temperature=0.4)
    dspy.settings.configure(lm=llamalm)
    

    #autolbl
    with open('config.json', 'r') as f:
        config = json.load(f)["defects"]
    gt_dataset = load_dataset(config['GT_IMAGES_DIRECTORY_PATH'], config['GT_ANNOTATIONS_DIRECTORY_PATH'], config['GT_DATA_YAML_PATH'])

    check_and_revise_prompt = dspy.Predict(CheckAndRevisePrompt)

    initial_prompt = "defect anomaly scratch crack split knot dead knot in wood"
    current_prompt = initial_prompt
    best_prompt = initial_prompt
    best_score = 0
    max_iter = 5

    wandb_prompt_table = wandb.Table(columns=["Iteration", "prompt","feedback", "class", "TP", "FP", "FN", "Accuracy", "F1"])
    wandb.log({"Prompt Iterations": wandb_prompt_table})
    df = pd.DataFrame(columns=["Iteration", "prompt","feedback", "class", "TP", "FP", "FN", "Accuracy", "F1"])
    for i in range(max_iter):
        print(f"Iteration {i+1} of {max_iter}")
        gt_class, TP, FP, FN, acc, F1  = label_images(config=config, gt_dataset=gt_dataset, prompt = current_prompt)
        #log metrics
        wandb.log(data={"iter": i, "TP": TP, "FP": FP, "FN": FN, "Accuracy": acc, "F1": F1})
        current_score = acc
        if current_score > best_score:
            best_prompt = current_prompt
            best_score = current_score
        result = check_and_revise_prompt(
            desired_score=1.0,
            current_score=current_score,
            current_prompt=current_prompt,
            context = "You are designing a prompt to help a vison model identify defects in wood images",
            best_prompt=best_prompt,
            best_score=best_score
        )
        print(f"Current score: {current_score}")
        if current_score != 1:
            current_prompt = result.revised_prompt
            #remove , from prompt
            current_prompt = current_prompt.replace(",", "").replace("-", "").replace(":", "").replace("\n", "")
            print(f"Feedback: {result.feedback}")
            print(f"Revised prompt: {result.revised_prompt}")
        else:
            print("Prompt is perfect")
            break
        #update pandas dataframe
        df = pd.concat([df, pd.DataFrame([{"Iteration": i+1, "prompt": current_prompt, "feedback": result.feedback, "class": gt_class, "TP": TP, "FP": FP, "FN": FN, "Accuracy": acc, "F1": F1}])], ignore_index=True)
        try:
            update_table_wandb("Prompt Iterations", [i, current_prompt, result.feedback, gt_class, TP, FP, FN, acc, F1])
        except Exception as e:
            wandb_prompt_table.add_data(i+1, current_prompt, result.feedback, gt_class, TP, FP, FN, acc, F1)
            print(f"Error updating wandb table: {e}")
        wandb.log({"Prompt Iterations": wandb_prompt_table})
        #log to pandas dataframe
        wandb_tab2 = wandb.Table(dataframe=df, allow_mixed_types=True)
        wandb.log({"Prompt Iterations2": wandb_tab2})

    print(f"Final prompt: {current_prompt}")
    wandb.finish()
if __name__ == "__main__":
    main()