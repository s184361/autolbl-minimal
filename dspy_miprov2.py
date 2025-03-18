import os
import json
import argparse
import gc

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

class Prompt_Design(dspy.Signature):
    """Design a prompt for the Vision Language Model to detect wood defects in an image."""

    task = dspy.InputField(
        desc="The input prompt to detect the defects in the image e.g. 'defect', 'crack', 'splinter'. Should be max 5 words."
    )
    vlm_prompt = dspy.OutputField(desc="Improved prompt for detecting defects in the image. Should be max 7 words.")


class DSPyPromptOptimizer:
    def __init__(self, config_path='config.json', section='defects', model='Florence', 
                 lm_model="ollama/deepseek-r1:1.5b", randomize=False):
        """
        Initialize the DSPy Prompt Optimizer
        
        Args:
            config_path: Path to the configuration file
            section: Section in the config file to use (e.g., 'defects', 'local')
            model: Vision model to use for detection
            lm_model: Language model to use with DSPy
            randomize: Whether to randomize initial prompts
        """
        self.config_path = config_path
        self.ds_name = section
        self.model = model
        self.randomize = randomize
        self.initial_prompt = "defect"  # Default initial prompt
        
        # Initialize wandb
        wandb.login()
        tags = ["dspy_prompt_optimization", self.ds_name, self.model,
                f"randomize={self.randomize}", f"[{self.initial_prompt}]"]
        
        self.run = wandb.init(project="dspy-wood-defects", 
                             config={
                                 "optimizer": "MIPROv2",
                                 "model": self.model,
                                 "dataset": self.ds_name,
                                 "randomize": self.randomize,
                                 "initial_prompt": self.initial_prompt
                             },
                             tags=tags)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup language model
        self.lm = dspy.LM(lm_model, api_base="http://localhost:11434", api_key="")
        dspy.configure(lm=self.lm)
        
        # Load ground truth dataset
        self.gt_dataset = self.load_gt_dataset()
        
        # Default prompts list
        self.default_prompts = [
            "blue stain", "crack", "dead knot", "knot missing", "knot with crack",
            "live knot", "marrow", "overgrown", "quartzity", "resin", "blue stain crack dead knot missing knot with crack live knot marrow overgrown quartzity resin"
        ]
        
        # Initialize prompt table for tracking results
        self.pd_prompt_table = pd.DataFrame(columns=[
            "prompt", "TP", "FP", "FN", "acc", "F1"
        ])
        
        # Define task description
        self.task_description = "Give a prompt for Vision Language Model to detect wood defects in an image. The output should be a string with no more than 100 characters."
        
        # Initialize optimization components
        self.program = None
        self.trainset = None
        self.devset = None
        self.optimized_program = None
        
        print(f"Initialized DSPyPromptOptimizer with model {self.model} for dataset {self.ds_name}")
        
    def load_gt_dataset(self):
        """Load ground truth dataset from config"""
        section_config = self.config.get(self.ds_name, self.config.get("local", {}))
        
        gt_dataset = load_dataset(
            section_config["GT_IMAGES_DIRECTORY_PATH"],
            section_config["GT_ANNOTATIONS_DIRECTORY_PATH"],
            section_config["GT_DATA_YAML_PATH"],
        )
        return gt_dataset
    
    def label_images(self, prompt: str, eval_metrics: bool = True):
        """
        Label images using the provided prompt and evaluate against ground truth.
        Returns metrics for the prompt's performance.
        """
        # Create the arguments
        args = argparse.Namespace(
            config=self.config_path,
            section=self.ds_name,
            model=self.model,
            tag="default",
            sahi=False,
            reload=False,
            ontology=f"{prompt}: defect",
            wandb=False,
            save_images=False,
        )

        # Run detection on images
        dataset = run_any_args(args)

        if eval_metrics:
            # Evaluate detections against ground truth
            confusion_matrix, acc, map_result = evaluate_detections(dataset, self.gt_dataset)
            
            print(f"Accuracy: {acc}")

            # Calculate metrics
            gt_class = "defect"
            TP = confusion_matrix[0, 0] / confusion_matrix.sum()
            FP = confusion_matrix[0, 1] / confusion_matrix.sum()
            FN = confusion_matrix[1, 0] / confusion_matrix.sum()
            F1 = 2 * TP / (2 * TP + FP + FN)
        else:
            gt_class = "defect"
            TP = None
            FP = None
            FN = None
            acc = None
            F1 = None
            
        return gt_class, TP, FP, FN, acc[0] if acc is not None else None, F1, dataset
    
    def metric_function(self, example, pred, trace=None):
        """
        Metric function to evaluate the output of the program.
        Returns the F1 score for the prompt.
        """
        print(f"Example: {example}")
        print(f"Prediction: {pred}")
        print(f"Trace: {trace}")
        
        # Handle different prediction formats
        if isinstance(pred, str):
            prompt = pred
        elif hasattr(pred, 'vlm_prompt'):
            prompt = pred.vlm_prompt
        elif hasattr(pred, 'output'):
            prompt = pred.output
        else:
            # Fallback for unknown prediction format
            print(f"WARNING: Unexpected prediction format: {type(pred)}")
            prompt = str(pred)

        # Free memory before running detection
        gc.collect()

        # Label images and get metrics
        _, TP, FP, FN, acc, F1, _ = self.label_images(prompt=prompt)

        # Log the metrics for this evaluation
        wandb.log({
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accuracy": acc,
            "F1": F1,
            "prompt": prompt
        })
        
        # Update prompt table
        new_row = pd.DataFrame({
            'prompt': [prompt], 
            'TP': [TP], 
            'FP': [FP], 
            'FN': [FN], 
            'acc': [acc], 
            'F1': [F1]
        })
        self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
        
        # Log updated table to wandb
        wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})

        # Free memory after running detection
        gc.collect()
        print(f"F1: {F1}")
        return F1
    
    def prepare_examples(self, example_file="prompt_examples_small.csv"):
        """
        Prepare examples for DSPy from existing data or by creating them
        """
        try:
            # Try to load existing examples
            prompt_examples = pd.read_csv(example_file)
            print(f"Loaded {len(prompt_examples)} prompt examples from {example_file}")
        except:
            print(f"Creating new prompt examples file {example_file}")
            # Create new examples if file doesn't exist
            prompt_examples = pd.DataFrame(columns=["prompt", "F1"])
            for prompt in self.default_prompts:
                _, TP, FP, FN, acc, F1, _ = self.label_images(prompt=prompt)
                prompt_examples = pd.concat([
                    prompt_examples,
                    pd.DataFrame([{
                        "prompt": prompt,
                        "F1": F1
                    }])
                ], ignore_index=True)
            # Save for future use
            prompt_examples.to_csv(example_file, index=False)
            
        return prompt_examples
    
    def get_dspy_examples(self, df, k):
        """
        Get balanced examples for DSPy from a dataframe
        """
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
                            task=self.task_description,
                            vlm_prompt=row["prompt"],
                        ).with_inputs("task")
                    )
            except:
                # If there aren't enough samples in a class, use what's available
                class_df = df[df['F1_class'] == f1_class]
                for _, row in class_df.iterrows():
                    dspy_examples.append(
                        dspy.Example(
                            task=self.task_description,
                            vlm_prompt=row["prompt"],
                        ).with_inputs("task")
                    )
        
        return dspy_examples
    
    def create_datasets(self, prompt_examples, train_k=10, dev_k=3):
        """
        Create training and development datasets with balanced examples
        """
        # Create balanced train and dev sets
        trainset = self.get_dspy_examples(prompt_examples, k=train_k)
        devset = self.get_dspy_examples(prompt_examples, k=dev_k)
        
        # Ensure no overlap between train and dev sets
        dev_prompts = set([ex.vlm_prompt for ex in devset])
        trainset = [ex for ex in trainset if ex.vlm_prompt not in dev_prompts]
        
        print(f"Created training set with {len(trainset)} examples and dev set with {len(devset)} examples")
        
        self.trainset = trainset
        self.devset = devset
        return trainset, devset
    
    def step(self, prompt):
        """
        Evaluate a single prompt and return metrics
        Similar to the step method in test_opt_ax.py
        """
        gc.collect()
        gt_class, TP, FP, FN, acc, F1, dataset = self.label_images(prompt=prompt)
        
        # Log to wandb
        wandb.log({
            "prompt": prompt,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accuracy": acc,
            "F1": F1
        })
        
        return gt_class, TP, FP, FN, acc, F1, dataset
    
    def evaluate_baseline(self):
        """
        Create and evaluate the baseline program
        """
        print("Evaluating baseline program...")
        self.program = dspy.Predict(Prompt_Design)
        
        evaluate = Evaluate(
            devset=self.devset,
            metric=self.metric_function,
            num_threads=1,
            display_progress=True,
            display_table=True,
            provide_traceback=True
        )
        
        baseline_metrics = evaluate(self.program)
        baseline_result = self.program(task=self.devset[0].task)
        baseline_prompt = baseline_result.vlm_prompt
        
        # Evaluate the baseline prompt directly
        _, TP, FP, FN, acc, F1, _ = self.step(baseline_prompt)
        
        # Log baseline results
        wandb.log({
            "prompt": baseline_prompt,
            "accuracy": acc,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "F1": F1
        })
        
        return baseline_metrics, baseline_result, F1
    
    def optimize_with_mipro(self):
        """
        Optimize the program with MIPROv2
        """
        print("Optimizing program with MIPROv2...")
        teleprompter = MIPROv2(
            metric=self.metric_function,
            auto="light",
            max_bootstrapped_demos=2,
            num_threads=1,
        )

        # Compile optimized program
        self.optimized_program = teleprompter.compile(
            self.program.deepcopy(),
            trainset=self.trainset,
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            requires_permission_to_run=False,
            minibatch=False
        )

        # Save optimized program for future use
        save_path = f"mipro_{self.ds_name}_optimized.json"
        self.optimized_program.save(save_path)
        print(f"Saved optimized program to {save_path}")
        
        return self.optimized_program
    
    def evaluate_optimized(self):
        """
        Evaluate the optimized program
        """
        print("Evaluating optimized program...")
        evaluate = Evaluate(
            devset=self.devset,
            metric=self.metric_function,
            num_threads=1,
            display_progress=True,
            display_table=True,
            provide_traceback=True
        )
        
        optimized_metrics = evaluate(self.optimized_program, devset=self.devset)
        
        # Generate and test the final prompt
        final_result = self.optimized_program(task=self.devset[0].task)
        final_prompt = final_result.vlm_prompt
        
        print(f"Final optimized prompt: {final_prompt}")
        
        # Final evaluation with the optimized prompt
        gc.collect()
        _, TP, FP, FN, acc, F1, _ = self.step(final_prompt)
        
        # Log final results
        final_metrics = {
            "prompt": final_prompt,
            "accuracy": acc,
            "F1": F1,
            "TP": TP,
            "FP": FP,
            "FN": FN
        }
        wandb.log(final_metrics)
        
        return final_prompt, final_metrics, F1
    
    def run_optimization(self, example_file="prompt_examples_small.csv"):
        """
        Run the full optimization pipeline
        """
        # Prepare examples
        prompt_examples = self.prepare_examples(example_file)
        
        # Create datasets
        self.create_datasets(prompt_examples)
        
        # Evaluate baseline
        baseline_metrics, baseline_result, baseline_f1 = self.evaluate_baseline()
        
        # Optimize with MIPROv2
        self.optimize_with_mipro()
        
        # Evaluate optimized program
        final_prompt, final_metrics, optimized_f1 = self.evaluate_optimized()
        
        # Compare baseline to optimized
        print(f"Baseline F1: {baseline_f1}")
        print(f"Optimized F1: {optimized_f1}")
        print(f"Improvement: {optimized_f1 - baseline_f1}")
        
        # Log comparison
        wandb.log({
            "baseline_F1": baseline_f1,
            "optimized_F1": optimized_f1,
            "improvement": optimized_f1 - baseline_f1
        })
        
        # Finish wandb run
        self.run.finish()
        
        return final_prompt, final_metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DSPy Prompt Optimization")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--section", default="defects", help="Section in config file")
    parser.add_argument("--model", default="DINO", help="Vision model to use")
    parser.add_argument("--lm_model", default="ollama/deepseek-r1:1.5b", help="Language model to use")
    parser.add_argument("--randomize", action="store_true", help="Whether to randomize initial prompts")
    args = parser.parse_args()
    
    # Create optimizer and run the optimization pipeline
    optimizer = DSPyPromptOptimizer(
        config_path=args.config,
        section=args.section,
        model=args.model,
        lm_model=args.lm_model,
        randomize=args.randomize
    )
    
    final_prompt, final_metrics = optimizer.run_optimization()
    return final_prompt, final_metrics


if __name__ == "__main__":
    main()
