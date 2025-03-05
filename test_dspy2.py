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
import random

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


class Prompt_Design(dspy.Signature):
    """Design a prompt for the Vision Language Model to detect wood defects in an image."""

    customer_message = dspy.InputField(
        desc="Customer message during customer service interaction"
    )
    intent_labels = dspy.InputField(desc="Labels that represent customer intent")
    answer = dspy.OutputField(desc="a label best matching customer's intent ")


def metric_function(example, pred):
    """
    Metric function to evaluate the output of the program.
    Returns the F1 score for the prompt.
    """
    # Make sure we can access the prompt field
    prompt = pred.answer if hasattr(pred, 'answer') else pred.output
    
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

    # Label images and get metrics
    _, TP, FP, FN, acc, F1 = label_images(
        config=config, gt_dataset=gt_dataset, prompt=prompt
    )
    
    # Free memory after running detection
    gc.collect()
    torch.cuda.empty_cache()
    print(f"F1: {F1}")
    return F1

def main():
    # Initialize wandb
    wandb.login()
    run = wandb.init(project="dspy-wood-defects", config={"optimizer": "MIPROv2"})

    # Setup language model
    lm = dspy.LM(
        "ollama/deepseek-r1:1.5b", api_base="http://localhost:11434", api_key=""
    )
    dspy.configure(lm=lm)

    # load prompt_examples.csv
    prompt_examples = pd.read_csv("prompt_examples.csv")
    dspy_examples = []
    customer_message = "Give a prompt for Vision Language Model to detect wood defects in an image.The ouput should be a string with no more than 100 characters."

    # Define function to get examples with balanced classes
    def get_dspy_examples(df, k):
        dspy_examples = []
        # Group by F1 score rounded to 1 decimal to create "classes"
        df['F1_class'] = df['F1'].round(1)
        classes = df['F1_class'].unique()
        
        for f1_class in classes:
            try:
                class_df = df[df['F1_class'] == f1_class].sample(n=k, replace=True)
                for _, row in class_df.iterrows():
                    dspy_examples.append(
                        dspy.Example(
                            customer_message=customer_message,
                            intent_labels=row["prompt"],
                            answer=row["F1"]
                        ).with_inputs("customer_message", "intent_labels")
                    )
            except:
                # If there aren't enough samples in a class, use what's available
                class_df = df[df['F1_class'] == f1_class]
                for _, row in class_df.iterrows():
                    dspy_examples.append(
                        dspy.Example(
                            customer_message=customer_message,
                            intent_labels=row["prompt"],
                            answer=row["F1"]
                        ).with_inputs("customer_message", "intent_labels")
                    )
        
        return dspy_examples
    
    # Create balanced train and dev sets
    trainset = get_dspy_examples(prompt_examples, k=10)
    devset = get_dspy_examples(prompt_examples, k=3)
    
    # Ensure no overlap between train and dev sets
    dev_prompts = set([ex.intent_labels for ex in devset])
    trainset = [ex for ex in trainset if ex.intent_labels not in dev_prompts]
    #remove dspy_examples from memory
    del dspy_examples, prompt_examples
    gc.collect()
    # Prepare results tracking
    df = pd.DataFrame(columns=[
        "Iteration", "prompt", "reasoning", "TP", "FP", "FN", "Accuracy", "F1"
    ])

    # Create and evaluate the baseline program
    print("Evaluating baseline program...")
    program = dspy.ChainOfThought(Prompt_Design)
    evaluate = Evaluate(
                    devset=devset,  # devset must be provided here
                    metric=metric_function, 
                    num_threads=1, 
                    display_progress=True, 
                    display_table=True,
                    provide_traceback=True
                )
    baseline_metrics = evaluate(program)

    # Log baseline results
    baseline_result = program(customer_message=devset[0].customer_message, intent_labels=devset[0].intent_labels)

    # Optimize with MIPROv2
    print("Optimizing program with MIPROv2...")
    teleprompter = MIPROv2(
        metric=metric_function,
        auto="light",
        max_bootstrapped_demos=0,
                )

    # Compile optimized program
    optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=2,
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
