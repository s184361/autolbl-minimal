import argparse
import json
import os
from run_any2 import run_any_args
from utils.check_labels import *
from transformers import BertTokenizerFast
import torch
from torch.nn import functional as F
from ultralytics.models.utils.loss import DETRLoss

import wandb
import pandas as pd
import numpy as np  # Ensure numpy is imported
import random
from utils.wandb_utils import compare_plot as compare_wandb_plot

from ax.service.ax_client import AxClient, ObjectiveProperties
class PromptOptimizer:
    def __init__(self, config_path='config.json', encoding_type='bert'):
        torch.cuda.empty_cache()
        wandb.login()
        self.randomize = True
        self.initial_prompt = "[PAD] knot [PAD] [PAD] defect [PAD] crack [PAD]"
        self.initial_prompt = "defect"
        self.initial_prompt = "[PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"
        self.model = "DINO"
        self.optimizer = "COBYLA"
        self.ds_name = "defects"
        self.encoding_type = encoding_type  # 'ascii' or 'bert'
        
        tags=["prompt_optimization_debug", self.ds_name, self.model, self.optimizer, 
              f"randomize={self.randomize}", f"[{self.initial_prompt}]", f"encoding={self.encoding_type}"]
        self.pd_prompt_table = pd.DataFrame()

        self.run = wandb.init(project="backprop", job_type="prompt_optimization", tags=tags)
        self.run.config.update({
            "model": self.model,
            "dataset": self.ds_name,
            "optimizer": self.optimizer,
            "maxiter": 100,
            "randomize": self.randomize,
            "initial_prompt": self.initial_prompt,
            "encoding_type": self.encoding_type
        })

        with open(config_path, 'r') as f:
            self.config = json.load(f)[self.ds_name]
            
        # Load ground truth dataset
        self.gt_dataset = load_dataset(
            self.config['GT_IMAGES_DIRECTORY_PATH'],
            self.config['GT_ANNOTATIONS_DIRECTORY_PATH'],
            self.config['GT_DATA_YAML_PATH']
        ) 
        self.gt_dataset = self.set_one_class(self.gt_dataset)
        self.check_classes(self.gt_dataset)
        self.gt_dict = {
                os.path.splitext(os.path.basename(image_path))[0] + ".jpg": (image_path, annotation)
                for image_path, _, annotation in self.gt_dataset
            }
        
        # Initialize tokenizer conditionally
        self.tokenizer = None
        if self.encoding_type == 'bert':
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        # Initialize with chosen encoding
        self.initial_prompt = "defect knot crack stain"
        self.input_ids = self.encode_prompt(self.initial_prompt)
        
        # If randomizing, modify values
        if self.randomize:
            if self.encoding_type == 'ascii':
                # Randomize ASCII values
                for i in range(random.randint(1, len(self.input_ids))):
                    self.input_ids[random.randint(0, len(self.input_ids)-1)] = random.randint(32, 126)
            else:
                # Randomize BERT token IDs
                for i in range(random.randint(1, len(self.input_ids))):
                    self.input_ids[random.randint(0, len(self.input_ids)-1)] = random.randint(0, len(self.tokenizer))
                
        self.ax_client = AxClient()
        print("Initial prompt:", self.decode_prompt(self.input_ids), self.input_ids)

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

    def encode_prompt(self, text):
        """Convert a string to token representation based on encoding type"""
        if self.encoding_type == 'ascii':
            return torch.tensor([ord(c) for c in text], dtype=torch.float32)
        else:
            # BERT tokenization
            input_ids = self.tokenizer(text)['input_ids']
            # Remove special tokens (first and last)
            return torch.tensor(input_ids[1:-1], dtype=torch.float32)

    def decode_prompt(self, token_ids, _=None):
        """Convert token IDs back to a string based on encoding type"""
        if self.encoding_type == 'ascii':
            # Round and clip to valid ASCII range (32-126 for printable chars)
            rounded = torch.round(token_ids).clamp(32, 126).to(torch.int64)
            # Convert ASCII values back to characters
            return ''.join([chr(int(code)) for code in rounded])
        else:
            # BERT decoding
            rounded = torch.round(token_ids).clamp(0, len(self.tokenizer)).to(torch.int64)
            return self.tokenizer.decode(rounded, skip_special_tokens=True)

    def step(self, prompt: str, eval_metrics: bool = False):
        args = argparse.Namespace(
            config='/zhome/4a/b/137804/Desktop/autolbl/config.json',
            section=self.ds_name,
            model=self.model,
            tag='default',
            sahi=False,
            reload=False,
            ontology=f'{prompt}: defect',
            wandb=False,
            save_images=False
        )
        dataset = run_any_args(args)
        if eval_metrics:
            confusion_matrix, acc, _ = evaluate_detections(dataset, self.gt_dataset)
            acc = acc[0]
            print(f"Accuracy: {acc}")
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
        #compare_wandb_plot(dataset, self.gt_dataset)
        return gt_class, TP, FP, FN, acc, F1, dataset

    def loss2(self, input_ids: torch.Tensor, eval_metrics: bool = False):
        prompt = self.decode_prompt(input_ids)
        print("Loss is being calculated for prompt:", prompt, "input_ids:", input_ids)
        gt_class, TP, FP, FN, acc, F1, dataset = self.step(prompt, eval_metrics)
        pred_bboxes = torch.empty(0, 4)
        pred_labels = torch.empty(0)
        gt_bboxes = torch.empty(0, 4)
        gt_labels = torch.empty(0)
        if self.gt_dict is None:
            self.gt_dict = {
                os.path.splitext(os.path.basename(image_path))[0] + ".jpg": (image_path, annotation)
                for image_path, _, annotation in self.gt_dataset
            }
        pred_scores = torch.empty(0, 2)
        gt_groups = []
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
            wandb.log({
                "loss_giou": 1000.0,
                "bbox loss": 1000.0,
                "class loss": 1000.0,
                "total loss": total_loss,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "accuracy": acc,
                "F1": F1,
                "prompt": prompt,
                "input_ids": input_ids,
                "error": "Empty tensors detected"
            })
            
            #update the prompt table
            new_row = pd.DataFrame({'prompt': [prompt], 'TP': [TP], 'FP': [FP], 'FN': [FN], 'acc': [acc], 'F1': [F1], 'error': ['Empty tensors']})
            self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
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
        total_loss = 5 * loss_output['loss_class'] + loss_output['loss_bbox'] / 1000 + loss_output['loss_giou']
        wandb.log({
            "loss_giou": loss_output["loss_giou"],
            "bbox loss": loss_output["loss_bbox"],
            "class loss": loss_output['loss_class'],
            "total loss": total_loss,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accuracy": acc,
            "F1": F1,
            "prompt": prompt,
            "input_ids": input_ids
        })

        #update the prompt table
        new_row = pd.DataFrame({'prompt': [prompt], 'TP': [TP], 'FP': [FP], 'FN': [FN], 'acc': [acc], 'F1': [F1]})
        self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
        #upload the prompt table to wandb
        wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})
        return gt_class, TP, FP, FN, acc, F1, dataset, total_loss, prompt

    def objective(self, x_np):
        x_tensor = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        gt_class, TP, FP, FN, acc, F1, dataset, loss, prompt = self.loss2(
            input_ids=x_tensor,
            eval_metrics=True
        )
        print(f"Loss: {loss}, Prompt: {prompt}")
        return float(loss)  # Ensure we return a Python float, not a tensor

    def optimize(self):
        # Define the length of our prompt
        prompt_length = len(self.input_ids)
        
        # Create parameters for each position in the prompt with appropriate bounds
        parameters = []
        if self.encoding_type == 'ascii':
            bounds = [32, 126]  # ASCII printable characters
        else:
            bounds = [0, len(self.tokenizer)]  # BERT vocabulary
            
        for i in range(prompt_length):
            parameters.append({
                "name": f"char_{i}",
                "type": "range",
                "bounds": bounds,
                "value_type": "int",
            })
        
        # Create experiment with individual parameters for each token position
        self.ax_client.create_experiment(
            name="prompt_optimization",
            parameters=parameters,
            objectives={"loss": ObjectiveProperties(minimize=True)},
        )

        # Create initial parameter dictionary with each token position
        initial_params = {}
        for i in range(prompt_length):
            initial_params[f"char_{i}"] = int(self.input_ids[i].item())
        
        # Attach the initial trial
        self.ax_client.attach_trial(parameters=initial_params)

        # Complete the initial trial
        self.ax_client.complete_trial(
            trial_index=0,
            raw_data={"loss": self.objective(self.input_ids)}
        )
        
        for i in range(self.run.config.maxiter):
            parameters, trial_index = self.ax_client.get_next_trial()
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=self.train_evaluate(parameters)
            )

    def train_evaluate(self, parameterization):
        # Convert the dictionary of individual character parameters back to a list
        x_np = []
        for i in range(len(parameterization)):
            x_np.append(parameterization[f"char_{i}"])
        
        # Make sure we return a native Python float, not a tensor
        loss_value = self.objective(x_np)
        return {"loss": float(loss_value)}



if __name__ == "__main__":
    optimizer = PromptOptimizer()
    optimizer.optimize()
    optimizer.ax_client.get_trials_data_frame()
