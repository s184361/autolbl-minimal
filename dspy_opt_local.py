import dspy
import os
from PIL import Image
from io import BytesIO

# from dotenv import load_dotenv

import json
import supervision as sv
from utils.check_labels import *
from run_any3 import run_any_args

import argparse
import wandb
from utils.wandb_utils import *
import pandas as pd
import os
import subprocess
import torch
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def label_images(config: None, gt_dataset: sv.DetectionDataset, prompt: str):
    # python run_any2.py --section defects --model Florence --ontology
    # Create the arguments
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    args = argparse.Namespace(
        config=config_path,
        section="local",
        model="DINO",
        tag="default",
        sahi=False,
        reload=False,
        ontology=f"{prompt}: defect",
        wandb=False,
        save_images=False,
        nms="no_nms",
        group=None,
    )
    dataset = run_any_args(args)

    confusion_matrix, precision, recall, F1, map50, map50_95 = evaluate_detections(dataset, gt_dataset)
    # ompare_plot(dataset, gt_dataset)
    # extract true positives
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {F1}")
    # return "class", "TP", "FP", "FN", "Accuracy", "F1"
    gt_class = "defect"
    TP = confusion_matrix[0, 0] / confusion_matrix.sum()
    FP = confusion_matrix[0, 1] / confusion_matrix.sum()
    FN = confusion_matrix[1, 0] / confusion_matrix.sum()
    
    # Handle arrays - take last value if multiple classes
    if len(F1) > 1:
        F1_score = F1[-1]
    else:
        F1_score = F1[0]
    
    if len(precision) > 1:
        precision_score = precision[-1]
    else:
        precision_score = precision[0]

    return gt_class, TP, FP, FN, precision_score, F1_score


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
    run = wandb.init(project="dspy", mode="offline")
    # load_dotenv()

    # Optional
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
    os.environ["ANTHROPIC_API_KEY"] = (
        "os.getenv("ANTHROPIC_API_KEY", "")"
    )
    try:
        process = subprocess.Popen(["ollama", "run", "deepseek-r1:1.5b"])
    except:
        process = subprocess.Popen(["/work3/s184361/ollama/bin/ollama", "run", "deepseek-r1:1.5b"])

    lm = dspy.LM(
        "ollama/deepseek-r1:1.5b", api_base="http://localhost:11434", api_key=""
    )
    dspy.configure(lm=lm)
    math = dspy.ChainOfThought("question -> answer: float")
    print(
        math(
            question="Two dice are tossed. What is the probability that the sum equals two?"
        )
    )
    # disconnect from the server
    # process.kill()
    # autolbl
    # Free GPU memory if possible
    gc.collect()
    torch.cuda.empty_cache()
    with open("config.json", "r") as f:
        config = json.load(f)["local"]
    gt_dataset = load_dataset(
        config["GT_IMAGES_DIRECTORY_PATH"],
        config["GT_ANNOTATIONS_DIRECTORY_PATH"],
        config["GT_DATA_YAML_PATH"],
    )

    check_and_revise_prompt = dspy.Predict(CheckAndRevisePrompt)

    initial_prompt = "defect"
    current_prompt = initial_prompt
    best_prompt = initial_prompt
    best_score = 0.03425
    max_iter = 5
    current_score = 0
    result = check_and_revise_prompt(
        desired_score=1.0,
        current_score=current_score,
        current_prompt=current_prompt,
        context="You are designing a prompt to help a vison model identify defects in wood images",
        best_prompt=best_prompt,
        best_score=best_score,
    )
    # wandb_prompt_table = wandb.Table(columns=["Iteration", "prompt","feedback", "class", "TP", "FP", "FN", "Accuracy", "F1"])
    # wandb.log({"Prompt Iterations": wandb_prompt_table})
    df = pd.DataFrame(
        columns=[
            "Iteration",
            "prompt",
            "feedback",
            "class",
            "TP",
            "FP",
            "FN",
            "Accuracy",
            "F1",
        ]
    )
    for i in range(max_iter):
        print(f"Iteration {i+1} of {max_iter}")
        gt_class, TP, FP, FN, acc, F1 = label_images(
            config=config, gt_dataset=gt_dataset, prompt=current_prompt
        )
        # Free GPU memory if possible
        gc.collect()
        torch.cuda.empty_cache()
        # log metrics
        wandb.log(
            data={"iter": i, "TP": TP, "FP": FP, "FN": FN, "Accuracy": acc, "F1": F1}
        )
        current_score = F1
        if current_score > best_score:
            best_prompt = current_prompt
            best_score = current_score
        # process = subprocess.Popen(["ollama", "run", "deepseek-r1:1.5b"])
        # dspy.configure(lm=lm)
        result = check_and_revise_prompt(
            desired_score=1.0,
            current_score=current_score,
            current_prompt=current_prompt,
            context="You are designing a prompt to help a vison model identify defects in wood images. Provide pompt with description of the wood defect, keywords or terminology for wood defects. Do not make your prompts too long. Giving commands to the vision model seems to be ineffective so do not use words like [find] or [detect] in your prompt. Remember you are trying to get current score as close to 1.0 as possible, so if you stop improving try to add novely to your prompt. Remember to keep the prompt short, up to 10 words. Mentioning that the prompt is detailed does not make detections better.",
            best_prompt=best_prompt,
            best_score=best_score,
        )
        print(f"Current score: {current_score}")
        if current_score != 1:
            current_prompt = result.revised_prompt
            # remove , from prompt
            current_prompt = (
                current_prompt.replace(",", "")
                .replace("-", "")
                .replace(":", "")
                .replace("\n", "")
            )
            print(f"Feedback: {result.feedback}")
            print(f"Revised prompt: {result.revised_prompt}")
        else:
            print("Prompt is perfect")
            break
        # update pandas dataframe
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Iteration": i + 1,
                            "prompt": current_prompt,
                            "feedback": result.feedback,
                            "class": gt_class,
                            "TP": TP,
                            "FP": FP,
                            "FN": FN,
                            "Accuracy": acc,
                            "F1": F1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        run.log({"prompt_iter": wandb.Table(dataframe=df)})
        # log to pandas dataframe
        wandb_tab2 = wandb.Table(dataframe=df)
        run.log({"prompt_iter2": wandb_tab2})
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Final prompt: {current_prompt}")
    run.finish()


if __name__ == "__main__":
    main()
