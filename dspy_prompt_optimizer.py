import os
import json
import argparse
import gc

import pandas as pd
import wandb
import dspy
import supervision as sv
import torch
from torch.nn import functional as F
from ultralytics.models.utils.loss import DETRLoss
from utils.check_labels import *
from run_any2 import run_any_args
from run_any3 import run_any_args as run_qwen
from utils.wandb_utils import *
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate 

# Set CUDA memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Prompt_Design(dspy.Signature):
    """Design a prompt for a detection task. Don't use words like 'detect' or 'find'.
    Don't ask for a specific design or look. Use only nouns and adjectives.
    Don't ask for a detailed or precise detection. Don't use words like 'image', 'photo', 'picture'.
    Don't as questions like 'what is the defect?' or 'where is the defect?'.
    Don't use words like 'improved', 'better', 'design', 'look','identify', 'locate', 'find', 'detect', 'image', 'photo', 'picture', 'detailed','precise'.
    """

    task = dspy.InputField(
        desc="Initial prompt of the defect user wants to detect. Should be short. It should not be a command like 'detect' or 'find'. It should be a noun or noun phrase. It should be a single word if possible. Use only nouns and adjectives. Do not use word like 'image', 'photo', 'picture', 'detailed','precise', 'detect', 'find'",
        default="defect",
    )
    vlm_prompt = dspy.OutputField(desc = f"Describe how does {task} look like. Here is an example 'Wooden surface with two circular holes. The holes appear to be made of a light-colored wood, possibly pine or birch. The wood has a rough texture, with visible knots and grooves. The edges of the wood are visible, and there is a small amount of dirt or grime on the surface. The image is taken from a top-down perspective, looking down on the holes.'")


class DSPyPromptOptimizer:
    def __init__(self, config_path='config.json', section='wood', model='DINO', 
                 lm_model="ollama/deepseek-r1:1.5b", randomize=False, use_detr_loss=True,loaded_model=None):
        """
        Initialize the DSPy Prompt Optimizer
        
        Args:
            config_path: Path to the configuration file
            section: Section in the config file to use (e.g., 'defects', 'local')
            model: Vision model to use for detection
            lm_model: Language model to use with DSPy
            randomize: Whether to randomize initial prompts
            use_detr_loss: Whether to use DETR loss function instead of F1 score
        """
        self.config_path = config_path
        self.ds_name = section
        self.model = model
        self.randomize = randomize
        self.initial_prompt = "defect"  # Default initial prompt
        self.use_detr_loss = use_detr_loss
        self.gt_dict = None  # Will be populated when needed for DETR loss
        self.loaded_model = loaded_model
        # Initialize wandb
        wandb.login()
        tags = ["dspy_prompt_optimization", self.ds_name, self.model,
                f"randomize={self.randomize}", f"[{self.initial_prompt}]",
                f"use_detr_loss={self.use_detr_loss}"]
        
        self.run = wandb.init(project="dspy-wood-defects", 
                             config={
                                 "optimizer": "MIPROv2",
                                 "model": self.model,
                                 "dataset": self.ds_name,
                                 "randomize": self.randomize,
                                 "initial_prompt": self.initial_prompt,
                                 "use_detr_loss": self.use_detr_loss
                             },
                             tags=tags)
        self.run_url = self.run.get_url()
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup language model - make sure server is running first
        self.ensure_ollama_server()
        self.lm = dspy.LM(lm_model, api_base="http://localhost:11434", api_key="")
        dspy.configure(lm=self.lm)
        
        # Load ground truth dataset
        self.gt_dataset = self.load_gt_dataset()
        self.gt_dataset = self.set_one_class(self.gt_dataset)
        self.check_classes(self.gt_dataset)
        # Default prompts list
        self.default_prompts = [
            "Wooden surface with two circular holes. The holes appear to be made of a light-colored wood, possibly pine or birch. The wood has a rough texture, with visible knots and grooves. The edges of the wood are visible, and there is a small amount of dirt or grime on the surface. The image is taken from a top-down perspective, looking down on the holes.",
            "blue. stain. crack. dead knot. missing knot. knot with crack. live knot. marrow. overgrown. quartzity. resin",
            "paint. hole. liqud. water. scratch",
            "broken small. broken large. contamination."
        ]
        
        # Initialize prompt table for tracking results
        columns = ["prompt", "TP", "FP", "FN", "acc", "F1"]
        if use_detr_loss:
            columns.extend(["total_loss", "loss_giou", "loss_bbox", "loss_class"])
            
        self.pd_prompt_table = pd.DataFrame(columns=columns)
        
        # Define task description
        self.task_description = ""
        
        # Initialize optimization components
        self.program = None
        self.trainset = None
        self.devset = None
        self.optimized_program = None

        print(f"Initialized DSPyPromptOptimizer with model {self.model} for dataset {self.ds_name}, DETR loss: {self.use_detr_loss}")
        
    def load_gt_dataset(self):
        """Load ground truth dataset from config"""
        section_config = self.config.get(self.ds_name, self.config.get("local", {}))
        
        gt_dataset = load_dataset(
            section_config["GT_IMAGES_DIRECTORY_PATH"],
            section_config["GT_ANNOTATIONS_DIRECTORY_PATH"],
            section_config["GT_DATA_YAML_PATH"],
        )
        return gt_dataset
    
    @staticmethod
    def set_one_class(gt_dataset):
        for key in gt_dataset.annotations.keys():
            gt_dataset.annotations[key].class_id = np.zeros_like(gt_dataset.annotations[key].class_id)
        gt_dataset.classes = ['defect']
        return gt_dataset

    @staticmethod
    def check_classes(gt_dataset):
        for key in gt_dataset.annotations.keys():
            for i in range(len(gt_dataset.annotations[key])):
                if gt_dataset.annotations[key][i].class_id != len(gt_dataset.classes) - 1:
                    return False
        return True

    def label_images(self, prompt: str, eval_metrics: bool = True):
        """
        Label images using the provided prompt and evaluate against ground truth.
        Returns metrics for the prompt's performance.
        """
        #remove "" from prompt
        prompt = prompt.replace('"','')
        
        # Create the arguments
        args = argparse.Namespace(
            config=self.config_path,
            section=self.ds_name,
            model=self.model,
            tag="default",
            sahi=False,
            reload=False,
            ontology=f"{prompt}",
            wandb=False,
            save_images=False,
        )

        # Run detection on images
        if self.model == "Qwen":
            dataset = run_qwen(args,loaded_model=self.loaded_model)
        else:
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
    
    def loss2(self, prompt: str, eval_metrics: bool = True):
        """
        Calculate DETR loss for the given prompt.
        Returns detailed metrics and loss values.
        """
        print(f"DETR Loss is being calculated for prompt: {prompt}")
        gc.collect()
        
        # Use step to get basic metrics and dataset
        gt_class, TP, FP, FN, acc, F1, dataset = self.step(prompt, eval_metrics)
        #Change to one class
        dataset = self.set_one_class(dataset)
        dataset.classes = ['defect']
        #check if the classes are set to one class
        if not self.check_classes(dataset):
            print("Error: Classes are not set to one class")
        # Initialize tensors
        pred_bboxes = torch.empty(0, 4)
        pred_labels = torch.empty(0)
        gt_bboxes = torch.empty(0, 4)
        gt_labels = torch.empty(0)
        
        # Create ground truth dictionary if needed
        if self.gt_dict is None:
            self.gt_dict = {
                os.path.splitext(os.path.basename(image_path))[0] + ".jpg": (image_path, annotation)
                for image_path, _, annotation in self.gt_dataset
            }
            
        # Initialize prediction scores and groups
        pred_scores = torch.empty(0, 2)
        gt_groups = []
        
        # Process each image in the dataset
        for image_path, _, annotation in dataset:
            name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
            if name_gt in self.gt_dict:
                pred_bboxes = torch.cat((pred_bboxes, torch.tensor(annotation.xyxy)))
                pred_labels = torch.cat((pred_labels, torch.tensor(annotation.class_id)))
                _, gt_annotation = self.gt_dict[name_gt]
                gt_bboxes = torch.cat((gt_bboxes, torch.tensor(gt_annotation.xyxy)))
                gt_labels = torch.cat((gt_labels, torch.tensor(gt_annotation.class_id)))
                if annotation.confidence is not None:
                    conf_tensor = torch.tensor(annotation.confidence, dtype=torch.float32).flatten()
                    conf_tensor = conf_tensor.unsqueeze(1)
                    score_tensor = torch.cat([1 - conf_tensor, conf_tensor], dim=1)
                    pred_scores = torch.cat((pred_scores, score_tensor), dim=0)

        # Check if we have empty tensors - if so, return a high default loss
        if len(pred_labels) == 0 or len(gt_labels) == 0:
            total_loss = torch.tensor(3000.0)  # High default loss value
            loss_giou = 1000.0
            loss_bbox = 1000.0
            loss_class = 1000.0
            
            wandb.log({
                "loss_giou": loss_giou,
                "bbox_loss": loss_bbox,
                "class_loss": loss_class,
                "total_loss": total_loss,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "accuracy": acc,
                "F1": F1,
                "prompt": prompt,
                "error": "Empty tensors detected"
            })
            
            # Update the prompt table
            new_row = pd.DataFrame({
                'prompt': [prompt], 
                'TP': [TP], 
                'FP': [FP], 
                'FN': [FN], 
                'acc': [acc], 
                'F1': [F1],
                'total_loss': [float(total_loss)],
                'loss_giou': [loss_giou],
                'loss_bbox': [loss_bbox],
                'loss_class': [loss_class],
                'error': ['Empty tensors'],
                "run_url": [wandb.Html(f"<a href='{self.run_url}'>{self.run.id}</a>")]})
            
            self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
            
            # Log updated table to wandb
            wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})
            
            return gt_class, TP, FP, FN, acc, F1, dataset, float(total_loss), prompt
            
        # Original code continues if tensors are not empty
        if len(pred_scores) == 0:
            pred_scores = torch.tensor([[-10, 10]])
            pred_scores = pred_scores.repeat(len(pred_labels), 1).float()
        if len(gt_groups) == 0:
            gt_groups = [len(gt_labels)]
            
        pred_labels = F.one_hot(pred_labels.long()).float()
        gt_labels = gt_labels.to(torch.int64)
        num_classes = len(self.gt_dataset.classes) + 1
        loss_fn = DETRLoss(nc=num_classes, aux_loss=False, use_fl=False, use_vfl=False)
        
        batch = {
            'cls': gt_labels.to(torch.int64),
            'bboxes': gt_bboxes.to(torch.float32),
            'gt_groups': gt_groups
        }
        
        pred_bboxes = pred_bboxes.unsqueeze(0).unsqueeze(0).to(torch.float32)
        pred_scores = pred_scores.unsqueeze(0).unsqueeze(0).to(torch.float32)
        
        loss_output = loss_fn.forward(
            pred_bboxes=pred_bboxes,
            pred_scores=pred_scores,
            batch=batch
        )
        
        loss_giou = float(loss_output["loss_giou"])
        loss_bbox = float(loss_output["loss_bbox"])
        loss_class = float(loss_output['loss_class'])
        total_loss = float(loss_class + loss_bbox / 1000 + loss_giou)
        
        wandb.log({
            "loss_giou": loss_giou,
            "bbox_loss": loss_bbox,
            "class_loss": loss_class,
            "total_loss": total_loss,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accuracy": acc,
            "F1": F1,
            "prompt": prompt
        })

        # Update the prompt table
        new_row = pd.DataFrame({
            'prompt': [prompt], 
            'TP': [TP], 
            'FP': [FP], 
            'FN': [FN], 
            'acc': [acc], 
            'F1': [F1],
            'total_loss': [total_loss],
            'loss_giou': [loss_giou],
            'loss_bbox': [loss_bbox],
            'loss_class': [loss_class]
        })
        
        self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
        print(self.pd_prompt_table)
        # Log updated table to wandb
        wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})
        print(f"Loss calculated: {total_loss}")
        return gt_class, TP, FP, FN, acc, F1, dataset, total_loss, prompt
    
    def metric_function(self, example, pred, trace=None):
        """
        Metric function to evaluate the output of the program.
        Returns either F1 score or negative DETR loss based on configuration.
        """
        # Ensure Ollama server is running before processing
        self.ensure_ollama_server()
        
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

        # Use appropriate loss function based on configuration
        if self.use_detr_loss:
            _, TP, FP, FN, acc, F1, _, loss, _ = self.loss2(prompt=prompt)
            metric_value = -loss  # Negative because we want to maximize the metric (minimize loss)
        else:
            # Label images and get metrics using F1 score
            _, TP, FP, FN, acc, F1, _ = self.label_images(prompt=prompt)
            metric_value = F1

        # Free memory after running detection
        gc.collect()
        
        # Return either F1 or negative loss as the optimization metric
        print(f"Metric value: {metric_value}")
        self.ensure_ollama_server()
        return metric_value
    
    def prepare_examples(self, example_file="prompt_examples_small.csv"):
        """
        Prepare examples for DSPy from existing data or by creating them
        
        Args:
            example_file: Path to CSV file containing prompt examples
            
        Returns:
            pd.DataFrame: DataFrame containing prompt examples with metrics
        """
        try:
            # Try to load existing examples with specific error handling
            prompt_examples = pd.read_csv(example_file)
            print(f"Loaded {len(prompt_examples)} prompt examples from {example_file}")
            
            # Validate loaded examples have required columns
            required_cols = ["prompt"] + (["F1", "total_loss"] if self.use_detr_loss else ["F1"])
            missing_cols = [col for col in required_cols if col not in prompt_examples.columns]
            if missing_cols:
                print(f"Warning: Missing required columns in {example_file}: {missing_cols}")
                raise ValueError(f"Invalid example file format, missing: {missing_cols}")
                
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            print(f"Creating new prompt examples file {example_file}: {str(e)}")
            # Create new examples if file doesn't exist or is invalid
            if self.use_detr_loss:
                columns = ["prompt", "F1", "total_loss"]
            else:
                columns = ["prompt", "F1"]
                    
            prompt_examples = pd.DataFrame(columns=columns)
            
            # Add progress tracking
            from tqdm import tqdm
            total_prompts = len(self.default_prompts)
            for i, prompt in enumerate(tqdm(self.default_prompts, desc="Evaluating prompts")):
                print(f"\nEvaluating prompt [{i+1}/{total_prompts}]: '{prompt}'")
                try:
                    # Try to evaluate this prompt
                    if self.use_detr_loss:
                        _, TP, FP, FN, acc, F1, _, loss, _ = self.loss2(prompt=prompt)
                        row_data = {
                            "prompt": prompt,
                            "F1": F1,
                            "total_loss": loss
                        }
                    else:
                        _, TP, FP, FN, acc, F1, _ = self.label_images(prompt=prompt)
                        row_data = {
                            "prompt": prompt,
                            "F1": F1
                        }
                    
                    # Add to DataFrame
                    prompt_examples = pd.concat([
                        prompt_examples,
                        pd.DataFrame([row_data])
                    ], ignore_index=True)
                    
                    # Save intermediate results every 5 prompts
                    if (i+1) % 5 == 0 or i == total_prompts-1:
                        prompt_examples.to_csv(f"{example_file}.partial", index=False)
                        
                except Exception as e:
                    print(f"Warning: Failed to evaluate prompt '{prompt}': {str(e)}")
            
            # Final validation before saving
            if len(prompt_examples) == 0:
                raise RuntimeError("No valid prompt examples were created")
                
            # Save final results
            prompt_examples.to_csv(example_file, index=False)
            print(f"Saved {len(prompt_examples)} prompt examples to {example_file}")
                
        return prompt_examples
    
    def get_dspy_examples(self, df, k):
        """
        Get balanced examples for DSPy from a dataframe, using either F1 score or total_loss
        """
        dspy_examples = []
        # Determine which metric to use (total_loss if available, otherwise F1)
        if 'total_loss' in df.columns and self.use_detr_loss:
            metric_col = 'total_loss'
            # mean
            df['metric_class'] = df[metric_col]
        else:
            metric_col = 'F1'
            # For F1, we group by F1 score rounded to 1 decimal
            df['metric_class'] = df[metric_col].round(1)
        
        classes = df['metric_class'].unique()
        
        for metric_class in classes:
            try:
                class_df = df[df['metric_class'] == metric_class].sample(n=k, replace=True)
                for _, row in class_df.iterrows():
                    dspy_examples.append(
                        dspy.Example(
                            task=row["prompt"],
                            vlm_prompt=row["prompt"],
                        ).with_inputs("task")
                    )
            except:
                # If there aren't enough samples in a class, use what's available
                class_df = df[df['metric_class'] == metric_class]
                for _, row in class_df.iterrows():
                    dspy_examples.append(
                        dspy.Example(
                            task=row["prompt"],
                            vlm_prompt=row["prompt"],
                        ).with_inputs("task")
                    )
        
        return dspy_examples
    
    def create_datasets(self, prompt_examples, train_k=2, dev_k=8):
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
        
        #self.trainset = trainset
        self.devset = devset
        self.trainset = devset
        return devset, devset
    
    def step(self, prompt, eval_metrics=True):
        """
        Evaluate a single prompt and return metrics
        Similar to the step method in test_opt_ax.py
        """
        gc.collect()
        gt_class, TP, FP, FN, acc, F1, dataset = self.label_images(prompt=prompt, eval_metrics=eval_metrics)
        
        # Log to wandb
        wandb.log({
            "step/prompt": prompt,
            "step/TP": TP,
            "step/FP": FP,
            "step/FN": FN,
            "step/accuracy": acc,
            "step/F1": F1
        })
        
        return gt_class, TP, FP, FN, acc, F1, dataset
    
    def evaluate_baseline(self):
        """
        Create and evaluate the baseline program
        """
        print("Evaluating baseline program...")
        self.program = dspy.Predict(Prompt_Design)
        self.program = dspy.ChainOfThought(Prompt_Design)
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
        if self.use_detr_loss:
            _, TP, FP, FN, acc, F1, _, loss, _ = self.loss2(prompt=baseline_prompt)
            wandb.log({
                "baseline/total_loss": loss,
                "baseline/prompt": baseline_prompt,
                "baseline/F1": F1,
                "baseline/accuracy": acc,
                "baseline/TP": TP,
                "baseline/FP": FP,
                "baseline/FN": FN
            })
            baseline_score = -loss
        else:
            _, TP, FP, FN, acc, F1, _ = self.step(baseline_prompt)
            wandb.log({
                "baseline/prompt": baseline_prompt,
                "baseline/F1": F1,
                "baseline/accuracy": acc,
                "baseline/TP": TP,
                "baseline/FP": FP,
                "baseline/FN": FN
            })
            baseline_score = F1
        
        return baseline_metrics, baseline_result, baseline_score
    
    def optimize_with_mipro(self):
        """
        Optimize the program with MIPROv2
        """
        print("Optimizing program with MIPROv2...")
        teleprompter = MIPROv2(
            metric=self.metric_function,
            auto="light",
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            num_threads=1
        )

        kwargs = dict(num_threads=64, display_progress=True, display_table=0) # Used in Evaluate class in the optimization process


        self.optimized_program = teleprompter.compile(
        self.program.deepcopy(),
        trainset=self.trainset,
        valset=self.devset,
        requires_permission_to_run=False,
        )

        # Save optimized program for future use
        mode = "detr_loss" if self.use_detr_loss else "f1"
        save_path = f"mipro_{self.ds_name}_{mode}_optimized.json"
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
        final_result = self.optimized_program(task="defect")
        final_prompt = final_result.vlm_prompt
        
        print(f"Final optimized prompt: {final_prompt}")
        
        # Final evaluation with the optimized prompt
        gc.collect()
        if self.use_detr_loss:
            _, TP, FP, FN, acc, F1, _, loss, _ = self.loss2(prompt=final_prompt)
            final_metrics = {
                "final/prompt": final_prompt,
                "final/total_loss": loss,
                "final/accuracy": acc,
                "final/F1": F1,
                "final/TP": TP,
                "final/FP": FP,
                "final/FN": FN
            }
            final_score = -loss
        else:
            _, TP, FP, FN, acc, F1, _ = self.step(final_prompt)
            final_metrics = {
                "final/prompt": final_prompt,
                "final/accuracy": acc,
                "final/F1": F1,
                "final/TP": TP,
                "final/FP": FP,
                "final/FN": FN
            }
            final_score = F1
            
        wandb.log(final_metrics)
        
        return final_prompt, final_metrics, final_score
    
    def run_optimization(self, example_file=None):
        """
        Run the full optimization pipeline
        """
        # Set example file name based on loss function choice
        if example_file is None:
            mode = "detr" if self.use_detr_loss else "f1"
            example_file = f"prompt_examples_{mode}_small.csv"
        
        # Prepare examples
        prompt_examples = self.prepare_examples(example_file)
        print("Prompt examples:")
        print(prompt_examples)
        # Create datasets
        self.create_datasets(prompt_examples)
        
        # Evaluate baseline
        baseline_metrics, baseline_result, baseline_score = self.evaluate_baseline()
        
        # Optimize with MIPROv2
        self.optimize_with_mipro()
        
        # Evaluate optimized program
        final_prompt, final_metrics, optimized_score = self.evaluate_optimized()
        
        # Compare baseline to optimized
        metric_name = "Loss" if self.use_detr_loss else "F1"
        improvement = optimized_score - baseline_score
        
        print(f"Baseline {metric_name}: {baseline_score}")
        print(f"Optimized {metric_name}: {optimized_score}")
        print(f"Improvement: {improvement}")
        
        # Log comparison
        wandb.log({
            "comparison/baseline_score": baseline_score,
            "comparison/optimized_score": optimized_score,
            "comparison/improvement": improvement,
            "comparison/metric": metric_name
        })
        
        # Finish wandb run
        self.run.finish()
        
        return final_prompt, final_metrics

    def ensure_ollama_server(self):
        """Check if Ollama server is running and restart if needed"""
        import requests
        import time
        import subprocess
        import os
        import signal
        
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                print("Ollama server is running properly")
                return True
        except requests.exceptions.ConnectionError:
            print("Ollama server is not responding, attempting to restart...")
        
        # Kill any existing Ollama processes
        try:
            process = subprocess.run(
                "pgrep -f '/work3/s184361/ollama/bin/ollama serve'", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            if process.stdout.strip():
                pids = process.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"Terminated Ollama process with PID {pid}")
                        except:
                            pass
        except Exception as e:
            print(f"Error killing existing Ollama processes: {e}")
        
        # Start a new Ollama server
        try:
            subprocess.Popen("/work3/s184361/ollama/bin/ollama serve", shell=True)
            print("Started new Ollama server")
            # Wait for server to start
            time.sleep(10)
            return True
        except Exception as e:
            print(f"Error starting Ollama server: {e}")
            return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DSPy Prompt Optimization")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--section", default="wood", help="Section in config file")
    parser.add_argument("--model", default="Qwen", help="Vision model to use")
    parser.add_argument("--lm_model", default="ollama/gemma3:1b", help="Language model to use")
    parser.add_argument("--randomize", default=False, type=bool, help="Randomize initial prompts")
    parser.add_argument("--use_detr_loss",default=True, type=bool, help="Use DETR loss function instead of F1 score")
    parser.add_argument("--example_file", default=None, help="Path to example file (optional)")
    args = parser.parse_args()
    if args.model == "Qwen":
        # Set up Qwen model
        from utils.qwen25_model import Qwen25VL
        from autodistill.detection import CaptionOntology
        q25 = Qwen25VL(ontology=CaptionOntology({"defect":"defect"}),hf_token="os.getenv("HF_TOKEN", "")")
    else:
        q25 = None
    # Create optimizer and run the optimization pipeline
    optimizer = DSPyPromptOptimizer(
        config_path=args.config,
        section=args.section,
        model=args.model,
        lm_model=args.lm_model,
        randomize=args.randomize,
        use_detr_loss=args.use_detr_loss,
        loaded_model=q25
    )
    
    final_prompt, final_metrics = optimizer.run_optimization(args.example_file)
    return final_prompt, final_metrics


if __name__ == "__main__":
    main()