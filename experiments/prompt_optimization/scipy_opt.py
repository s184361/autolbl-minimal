import argparse
import json
import os
from autolbl.cli.infer import run_any_args
from autolbl.evaluation.metrics import *
from transformers import BertTokenizerFast
import torch
from torch.nn import functional as F

# Try to import DETRLoss, but provide fallback if not available
try:
    from ultralytics.models.utils.loss import DETRLoss
    DETR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        # Try alternative import path for different ultralytics versions
        from ultralytics.yolo.utils.loss import DETRLoss
        DETR_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        print("Warning: DETRLoss not available in this ultralytics version. DETR loss calculation will be disabled.")
        DETR_AVAILABLE = False
        DETRLoss = None

import wandb
from scipy.optimize import minimize, differential_evolution
import numpy as np
import random
from autolbl.visualization.wandb import compare_plot as compare_wandb_plot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--randomize', type=bool, default=False)
    parser.add_argument('--initial_prompt', type=str, default='[PAD] [PAD] [PAD]')
    parser.add_argument('--ds_name', type=str, default='local_wood')
    parser.add_argument('--model', type=str, default='DINO')
    parser.add_argument('--optimizer', type=str, choices=['COBYLA', 'differential_evolution'], default='differential_evolution')
    parser.add_argument('--encoding_type', type=str, choices=['bert', 'ascii'], default='bert')
    parser.add_argument('--maxiter', type=int, default=2)
    parser.add_argument('--obj', type=str, default='loss')
    parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline', 'disabled'], default='online',
                        help='WandB mode: online (requires login), offline (local logging), disabled (no wandb)')
    return parser.parse_args()

class PromptOptimizer:
    def __init__(self,
                config_path='config.json',
                encoding_type='bert',
                randomize=False,
                model="DINO",
                optimizer="differential_evolution",
                ds_name="wood",
                initial_prompt="[PAD] [PAD] [PAD]",
                maxiter=100,
                obj='loss',
                wandb_mode='online'):
        
        torch.cuda.empty_cache()
        
        # Store wandb mode
        self.wandb_mode = wandb_mode
        
        # Only login if using online mode
        if wandb_mode == 'online':
            wandb.login()
        
        # Store configuration
        self.randomize = randomize
        self.initial_prompt = initial_prompt
        self.model = model
        self.optimizer = optimizer
        self.ds_name = ds_name
        self.encoding_type = encoding_type
        self.maxiter = maxiter
        self.obj = obj
        self.config_path = config_path  # Store config path for later use
        # Initialize wandb
        tags = [
            "prompt_optimization", 
            self.ds_name, 
            self.model, 
            self.optimizer, 
            f"randomize={self.randomize}", 
            f"encoding={self.encoding_type}"
        ]
        self.pd_prompt_table = pd.DataFrame()

        if self.wandb_mode == 'disabled':
            # Mock wandb run object for disabled mode
            class MockRun:
                def __init__(self):
                    self.id = 'local'
                    self.url = 'disabled'
                    self.config = type('obj', (object,), {'update': lambda x: None})()
                def log(self, data, step=None):
                    pass
            self.run = MockRun()
        else:
            # Initialize wandb with specified mode
            self.run = wandb.init(
                project="merged_prompt_opt", 
                job_type=self.optimizer, 
                tags=tags, 
                group=self.ds_name,
                name=f"{self.model}_{self.encoding_type}",
                mode=self.wandb_mode  # 'online' or 'offline'
            )
        
        if self.wandb_mode != 'disabled':
            self.run.config.update({
                "model": self.model,
                "dataset": self.ds_name,
                "optimizer": self.optimizer,
                "maxiter": self.maxiter,
            "randomize": self.randomize,
            "initial_prompt": self.initial_prompt,
            "encoding_type": self.encoding_type,
            "objective": self.obj
        })

        # Load dataset config
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
        
        # Initialize prompt with chosen encoding
        self.input_ids = self.encode_prompt(self.initial_prompt)
        
        # If randomizing, modify values based on encoding type
        if self.randomize:
            if self.encoding_type == 'ascii':
                # Randomize ASCII values
                for i in range(random.randint(1, len(self.input_ids))):
                    self.input_ids[random.randint(0, len(self.input_ids)-1)] = random.randint(32, 126)
            else:
                # Randomize BERT token IDs
                for i in range(random.randint(1, len(self.input_ids))):
                    self.input_ids[random.randint(0, len(self.input_ids)-1)] = random.randint(0, len(self.tokenizer))
                # Set the middle entry to the token for "defect" (if using BERT)
                if len(self.input_ids) > 1:
                    self.input_ids[len(self.input_ids) // 2] = 21262  # "defect" token in BERT
        
        self.run_url = f"https://wandb.ai/{self.run.entity}/{self.run.project}/runs/{self.run.id}"
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

    def decode_prompt(self, token_ids):
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
            config=self.config_path,
            section=self.ds_name,
            model=self.model,
            tag='default',
            sahi=False,
            reload=False,
            ontology=f'{prompt}: defect',
            wandb=False,
            save_images=False,
            nms="no_nms",
            group=None
        )
        dataset = run_any_args(args)
        if eval_metrics:
            confusion_matrix, precision, recall, F1, map05, map05095 = evaluate_detections(dataset= dataset, gt_dataset=self.gt_dataset, log_wandb=False)
            print(f"Precision: {precision}, Recall: {recall}, F1: {F1}")
            gt_class = "defect"
        else:
            gt_class = "defect"
            precision = None
            recall = None
            F1 = None
            map05 = None
            map05095 = None
            
        return gt_class, precision, recall, F1, map05, map05095, dataset

    def loss2(self, input_ids: torch.Tensor, eval_metrics: bool = False):
        prompt = self.decode_prompt(input_ids)
        print(f"Evaluating prompt: '{prompt}'")
        
        gt_class, precision, recall, F1, map05, map05095, dataset = self.step(prompt, eval_metrics)
        
        # Initialize tensors
        pred_bboxes = torch.empty(0, 4)
        pred_labels = torch.empty(0)
        gt_bboxes = torch.empty(0, 4)
        gt_labels = torch.empty(0)
        
        # Process each image in the dataset
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

        # Check if we have empty tensors
        if len(pred_labels) == 0 or len(gt_labels) == 0:
            total_loss = torch.tensor(3000.0)  # High default loss value
            loss_giou = 1000.0
            loss_bbox = 1000.0
            loss_class = 1000.0
            
            # Log metrics
            wandb.log({
                "loss_giou": loss_giou,
                "bbox_loss": loss_bbox,
                "class_loss": loss_class,
                "total_loss": total_loss,
                "precision": precision[0] if precision is not None else None,
                "recall": recall[0] if recall is not None else None,
                "F1": F1[0] if F1 is not None else None,
                "map05": map05,
                "map05095": map05095,
                "prompt": prompt,
                "error": "Empty tensors detected"
            })
            
            # Update prompt table
            new_row = pd.DataFrame({
                'prompt': [prompt], 
                'precision': [precision[0] if precision is not None else None], 
                'recall': [recall[0] if recall is not None else None],
                'F1': [F1[0] if F1 is not None else None],
                'map05': [map05],
                'map05095': [map05095],
                'total_loss': [float(total_loss)],
                'loss_giou': [loss_giou],
                'loss_bbox': [loss_bbox],
                'loss_class': [loss_class],
                'optimized' : [False],
                "run_url": [wandb.Html(f"<a href='{self.run.url}'>{self.run.id}</a>")]
            })
            
            self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
            wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})
            
            return gt_class, precision, recall, F1, map05, map05095, dataset, float(total_loss), prompt
            
        # Continue if tensors are not empty
        if len(pred_scores) == 0:
            pred_scores = torch.tensor([[-10, 10]])
            pred_scores = pred_scores.repeat(len(pred_labels), 1).float()
        if len(gt_groups) == 0:
            gt_groups = [len(gt_labels)]
            
        pred_labels = F.one_hot(pred_labels.long()).float()
        gt_labels = gt_labels.to(torch.int64)
        num_classes = len(self.gt_dataset.classes) + 1
        
        batch = {
            'cls': gt_labels,
            'bboxes': gt_bboxes.to(torch.float32),
            'gt_groups': gt_groups
        }
        
        pred_bboxes = pred_bboxes.unsqueeze(0).unsqueeze(0).to(torch.float32)
        pred_scores = pred_scores.unsqueeze(0).unsqueeze(0).to(torch.float32)
        
        # Calculate DETR loss if available
        if DETR_AVAILABLE and DETRLoss is not None:
            loss_fn = DETRLoss(nc=num_classes, aux_loss=False, use_fl=False, use_vfl=False)
            loss_output = loss_fn.forward(
                pred_bboxes=pred_bboxes,
                pred_scores=pred_scores,
                batch=batch
            )
            total_loss = 5 * loss_output['loss_class'] + loss_output['loss_bbox'] / 1000 + loss_output['loss_giou']
            loss_giou = loss_output["loss_giou"]
            loss_bbox = loss_output["loss_bbox"]
            loss_class = loss_output['loss_class']
        else:
            # Fallback: use F1 score as loss metric
            total_loss = 1.0 - F1  # Loss is inverse of F1 score
            loss_giou = 0.0
            loss_bbox = 0.0
            loss_class = total_loss
        
        # Log results
        wandb.log({
            "loss_giou": loss_giou,
            "bbox_loss": loss_bbox,
            "class_loss": loss_class,
            "total_loss": total_loss,
            "precision": precision[0] if precision is not None else None,
            "recall": recall[0] if recall is not None else None,
            "F1": F1[0] if F1 is not None else None,
            "map05": map05,
            "map05095": map05095,
            "prompt": prompt
        })

        # Update prompt table
        new_row = pd.DataFrame({
            'prompt': [prompt], 
            'precision': [precision[0] if precision is not None else None], 
            'recall': [recall[0] if recall is not None else None],
            'F1': [F1[0] if F1 is not None else None],
            'map05': [map05],
            'map05095': [map05095],
            'total_loss': [total_loss.item()],
            'loss_giou': [loss_output["loss_giou"].item()],
            'loss_bbox': [loss_output["loss_bbox"].item()],
            'loss_class': [loss_output['loss_class'].item()],
            'optimized': [False],
        })
            
        self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
        wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})
        
        return gt_class, precision, recall, F1, map05, map05095, dataset, total_loss, prompt

    def objective(self, x_np):
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        gt_class, precision, recall, F1, map05, map05095, dataset, loss, prompt = self.loss2(
            input_ids=x_tensor,
            eval_metrics=True
        )
        #choose objective based on config
        if self.obj == 'loss':
            print(f"Loss: {loss}, Prompt: {prompt}")
            return loss
        elif self.obj == 'F1':
            print(f"F1: {F1}, Prompt: {prompt}")
            return -F1[0] if F1 is not None else 0.0

    def optimize(self):
        # Create initial guess from prompt tensor
        x0 = self.input_ids.detach().numpy().flatten()
        
        # Set bounds based on encoding type
        if self.encoding_type == 'ascii':
            bounds = [(32, 126)] * len(x0)  # ASCII printable chars
        else:
            bounds = [(0, len(self.tokenizer))] * len(x0)  # BERT vocab
        
        # Choose optimizer based on config
        if self.optimizer == "COBYLA":
            result = minimize(
                self.objective,
                x0,
                method="COBYLA", 
                bounds=bounds,
                options={'maxiter': self.maxiter}
            )
        elif self.optimizer == "differential_evolution":
            result = differential_evolution(
                self.objective,
                bounds,
                popsize=15,
                mutation=(0.5, 1.5),
                recombination=0.7,
                maxiter=self.maxiter
            )
            
        # Process results
        optimized_x = result.x
        optimized_tensor = torch.tensor(optimized_x, dtype=torch.float32)
        optimized_prompt = self.decode_prompt(optimized_tensor)
        
        print("Optimized prompt:", optimized_prompt)
        
        # Log final results
        wandb.log({
            "optimized_prompt": optimized_prompt,
            "final_loss": result.fun
        })
        
        # Run final evaluation with optimized prompt
        gt_class, precision, recall, F1, map05, map05095, dataset, loss, prompt = self.loss2(optimized_tensor, True)
                # Update prompt table
        new_row = pd.DataFrame({
            'prompt': [prompt], 
            'precision': [precision[0] if precision is not None else None], 
            'recall': [recall[0] if recall is not None else None],
            'F1': [F1[0] if F1 is not None else None],
            'map05': [map05],
            'map05095': [map05095],
            'total_loss': [float(loss)],
            'loss_giou': [None],
            'loss_bbox': [None],
            'loss_class': [None],
            'optimized' : [False],
            "run_url": [wandb.Html(f"<a href='{self.run.url}'>{self.run.id}</a>")]
        })
        
        self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
        wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})
        #if len(dataset)<100:
            #compare_wandb_plot(dataset, self.gt_dataset)
        
        # Finish wandb run
        self.run.finish()


if __name__ == "__main__":
    args = parse_args()
    
    optimizer = PromptOptimizer(
        config_path=args.config,
        encoding_type=args.encoding_type,
        randomize=args.randomize,
        model=args.model,
        optimizer=args.optimizer,
        ds_name=args.ds_name,
        initial_prompt=args.initial_prompt,
        maxiter=args.maxiter,
        obj=args.obj,
        wandb_mode=args.wandb_mode
    )
    
    optimizer.optimize()
    
    # Clean up GPU memory
    torch.cuda.empty_cache()
