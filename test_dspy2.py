import os
import json
import argparse
import gc
import torch
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image
import requests
import subprocess
import pandas as pd
import wandb
import dspy
import supervision as sv
from utils.check_labels import *
from run_any2 import run_any_args
from utils.wandb_utils import *
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

# Set CUDA memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def label_images(config: None, gt_dataset: sv.DetectionDataset, prompt: str):
    """
    Label images using the provided prompt and evaluate against ground truth.
    Returns metrics for the prompt's performance.
    """
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
    )

    # Run detection on images
    dataset = run_any_args(args)

    # Evaluate detections against ground truth
    confusion_matrix, acc, map_result = evaluate_detections(dataset, gt_dataset)

    print(f"Accuracy: {acc}")

    # Calculate metrics
    gt_class = "defect"
    TP = confusion_matrix[0, 0] / confusion_matrix.sum()
    FP = confusion_matrix[0, 1] / confusion_matrix.sum()
    FN = confusion_matrix[1, 0] / confusion_matrix.sum()
    F1 = 2 * TP / (2 * TP + FP + FN)

    return gt_class, TP, FP, FN, acc[0], F1

class PromptGeneration(dspy.Signature):
    """
    Give a prompt for Vision Language Model to detect wood defects in an image.
    The ouput should be a string with no more than 100 characters.
    """
    task = dspy.InputField(description="Task description for detecting wood defects in an image.")
    prompt = dspy.OutputField(description="Prompt for detecting wood defects in an image.")

class PromptTask(dspy.Module):
    """
        DSPy module to optimize prompts for detecting wood defects.
        """
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(PromptGeneration)
    
    def forward(self):
        # Forward now expects task_description to match the signature
        return self.prog(task_description="Detect wood defects in an image.")

def metric_function(example, pred):
    """
    Metric function to evaluate the output of the program.
    Returns the F1 score for the prompt.
    """
    # Make sure we can access the prompt field
    prompt = pred.prompt if hasattr(pred, 'prompt') else pred.output
    
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)["local"]

    # Load ground truth dataset
    gt_dataset = load_dataset(
        config["GT_IMAGES_DIRECTORY_PATH"],
        config["GT_ANNOTATIONS_DIRECTORY_PATH"],
        config["GT_DATA_YAML_PATH"],
    )

    # Free memory before running detection
    gc.collect()
    torch.cuda.empty_cache()

    try:
        # Label images and get metrics
        _, TP, FP, FN, acc, F1 = label_images(
            config=config, gt_dataset=gt_dataset, prompt=prompt
        )
        
        # Free memory after running detection
        gc.collect()
        torch.cuda.empty_cache()
        
        return F1
    except Exception as e:
        print(f"Error in metric function: {e}")
        return 0.0

def main():
    # Initialize wandb
    wandb.login()
    run = wandb.init(project="dspy-wood-defects", config={"optimizer": "MIPROv2"})

    # Setup language model
    lm = dspy.LM(
        "ollama/deepseek-r1:1.5b", api_base="http://localhost:11434", api_key=""
    )
    dspy.configure(lm=lm)

    # Define train examples - properly formatted for DSPy
    trainset = [
        dspy.Example(taks = "Give a prompt for Vision Language Model to detect wood defects in an image.The ouput should be a string with no more than 100 characters.",prompt="defect", score=0.035).with_inputs("prompt")
    ]

    # Define dev examples - properly formatted for DSPy
    devset = [dspy.Example(taks = "Give a prompt for Vision Language Model to detect wood defects in an image.The ouput should be a string with no more than 100 characters.",prompt="defect", score=0.035).with_inputs("prompt")]

    # Prepare results tracking
    df = pd.DataFrame(columns=[
        "Iteration", "prompt", "reasoning", "TP", "FP", "FN", "Accuracy", "F1"
    ])

    # Create and evaluate the baseline program
    print("Evaluating baseline program...")
    program = PromptTask()
    evaluate = Evaluate(
                    devset=devset,  # devset must be provided here
                    metric=metric_function, 
                    num_threads=1, 
                    display_progress=True, 
                    display_table=True
                )
    baseline_metrics = evaluate(program)

    # Log baseline results
    baseline_result = program()
    print(f"Baseline prompt: {baseline_result.output}")

    # Optimize with MIPROv2
    print("Optimizing program with MIPROv2...")
    teleprompter = MIPROv2(
        metric=metric_function,
        auto="light",
    )

    # Compile optimized program
    optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
        requires_permission_to_run=False,
        minibatch=False
    )

    # Save optimized program for future use
    optimized_program.save("mipro_wood_defect_optimized")

    # Evaluate the optimized program
    print("Evaluating optimized program...")
    optimized_metrics = evaluate(optimized_program, devset=devset)

    # Generate and test the final prompt
    final_result = optimized_program(task_description=devset[0].task_description)
    final_prompt = final_result.output

    print(f"Final optimized prompt: {final_prompt}")

    # Load config and dataset for final evaluation
    with open("config.json", "r") as f:
        config = json.load(f)["local"]

    gt_dataset = load_dataset(
        config["GT_IMAGES_DIRECTORY_PATH"],
        config["GT_ANNOTATIONS_DIRECTORY_PATH"],
        config["GT_DATA_YAML_PATH"],
    )

    # Final evaluation with the optimized prompt
    gc.collect()
    torch.cuda.empty_cache()
    gt_class, TP, FP, FN, acc, F1 = label_images(
        config=config, gt_dataset=gt_dataset, prompt=final_prompt
    )

    # Log final results
    final_metrics = {
        "final_prompt": final_prompt,
        "final_accuracy": acc,
        "final_F1": F1,
        "final_TP": TP,
        "final_FP": FP,
        "final_FN": FN
    }
    wandb.log(final_metrics)

    # Add to dataframe
    df = pd.concat([
        df,
        pd.DataFrame([{
            "Iteration": "Final",
            "prompt": final_prompt,
            "reasoning": "MIPROv2 optimized",
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Accuracy": acc,
            "F1": F1
        }])
    ], ignore_index=True)

    # Log table to wandb
    wandb.log({"results": wandb.Table(dataframe=df)})

    # Compare baseline to optimized
    print(f"Baseline F1: {baseline_metrics[0]}")
    print(f"Optimized F1: {F1}")
    print(f"Improvement: {F1 - baseline_metrics[0]}")

    run.finish()

if __name__ == "__main__":
    main()
