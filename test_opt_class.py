import argparse
import json
import os
from run_any2 import run_any_args
from utils.check_labels import *
from transformers import BertTokenizerFast
import torch
from torch.nn import functional as F
from ultralytics.models.utils.loss import DETRLoss
from tqdm import tqdm
import wandb
from scipy.optimize import minimize
import numpy as np  # Ensure numpy is imported
import random
class PromptOptimizer:
    def __init__(self, config_path='config.json'):
        torch.cuda.empty_cache()
        wandb.login()
        self.run = wandb.init(project="backprop")
        self.pd_prompt_table = pd.DataFrame()
        self.ds_name = "bottle"
        with open(config_path, 'r') as f:
            self.config = json.load(f)[self.ds_name]
        # Load ground truth dataset (assumes load_dataset is available via utils.check_labels)
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
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        # Initialize prompt tensor from a default prompt.
        input_ids = self.tokenizer("[PAD] knot [PAD] [PAD] defect [PAD] crack [PAD]")['input_ids']
        self.input_ids = torch.tensor(input_ids, dtype=torch.float32)
        #remove exclude fist and last token
        self.input_ids = self.input_ids[1:-1]
        #set a random entry to a random integer within the tokenizer's vocabulary
        for i in range(random.randint(1, len(self.input_ids))):
            self.input_ids[random.randint(0, len(self.input_ids)-1)] = random.randint(0, len(self.tokenizer))
        #set the middle entry to the token for "defect" with value 21262
        self.input_ids[len(self.input_ids) // 2] = 21262
        print("Initial prompt:", self.decode_prompt(self.tokenizer, self.input_ids), self.input_ids)

    @staticmethod
    def set_one_class(gt_dataset):
        for key in gt_dataset.annotations.keys():
            new_annotation = gt_dataset.annotations[key]
            new_annotation.class_id = np.zeros_like(new_annotation.class_id)
            gt_dataset.annotations[key] = new_annotation
        gt_dataset.classes = ['defect']
        return gt_dataset

    @staticmethod
    def check_classes(gt_dataset):
        for key in gt_dataset.annotations.keys():
            for i in range(len(gt_dataset.annotations[key])):
                if gt_dataset.annotations[key][i].class_id != len(gt_dataset.classes) - 1:
                    return False
        return True
    @staticmethod
    def decode_prompt(tokenizer, input_ids):
        rounded_prompt = torch.round(input_ids)
        rounded_prompt = torch.clamp(rounded_prompt, 0, len(tokenizer)).to(torch.int64)
        prompt = tokenizer.decode(rounded_prompt, skip_special_tokens=True)
        return prompt

    def step(self, prompt: str, eval_metrics: bool = False):
        args = argparse.Namespace(
            config='/zhome/4a/b/137804/Desktop/autolbl/config.json',
            section=self.ds_name,
            model='DINO',
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
        return gt_class, TP, FP, FN, acc, F1, dataset

    def loss2(self, input_ids: torch.Tensor, eval_metrics: bool = False):
        prompt = self.decode_prompt(self.tokenizer, input_ids)
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
            'cls': gt_labels,
            'bboxes': gt_bboxes,
            'gt_groups': gt_groups
        }
        pred_bboxes = pred_bboxes.unsqueeze(0).unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0).unsqueeze(0)
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
        return loss

    def optimize(self):
        # Create an initial guess from the prompt tensor.
        x0 = self.input_ids.detach().numpy().flatten()
        bounds = [(0, len(self.tokenizer))] * len(x0)
        result = minimize(self.objective, x0, jac=False, options={'maxiter': 100},method="COBYLA", bounds=bounds)
        optimized_x = result.x
        optimized_tensor = torch.tensor(optimized_x, dtype=torch.float32)
        optimized_prompt = self.decode_prompt(self.tokenizer, optimized_tensor)
        print("Optimized prompt:", optimized_prompt)
        wandb.log({
            "optimized_prompt": optimized_prompt,
            "final_loss": result.fun
        })
        self.run.finish()

if __name__ == "__main__":
    optimizer = PromptOptimizer()
    optimizer.optimize()
