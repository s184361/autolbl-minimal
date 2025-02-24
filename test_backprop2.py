import argparse
import json
import os
from run_any2 import run_any_args
import supervision as sv
from utils.check_labels import *
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from transformers import BertTokenizerFast
import torch
import open_clip
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss
from torchvision import datasets, ops
from ultralytics.models.utils.loss import DETRLoss
from tqdm import tqdm
import wandb
from scipy.optimize import minimize

def set_one_class(gt_dataset):

    for key in gt_dataset.annotations.keys():
        for i in range(len(gt_dataset.annotations[key])):
            gt_dataset.annotations[key][i].class_id = np.zeros_like(gt_dataset.annotations[key][i].class_id)
    gt_dataset.classes = ['defect']
    return gt_dataset

def loss2(config: None, gt_dataset: sv.DetectionDataset, input_ids: torch.Tensor, tokenizer: BertTokenizerFast, eval_metrics: bool = False):
    prompt = decode_prompt(tokenizer, input_ids)
    gt_class, TP, FP, FN, acc, F1, dataset = step(config, gt_dataset, prompt, eval_metrics)
    pred_bboxes = torch.empty(0, 4)
    pred_labels = torch.empty(0)
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.empty(0)
    gt_dict = {os.path.splitext(os.path.basename(image_path))[0] + ".jpg": (image_path, annotation) for image_path, _, annotation in gt_dataset}
    pred_scores = torch.empty(0, 2)
    gt_groups = []
    for image_path, _, annotation in dataset:
        name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        if name_gt in gt_dict:
            pred_bboxes = torch.cat((pred_bboxes, torch.tensor(annotation.xyxy)))
            pred_labels = torch.cat((pred_labels, torch.tensor(annotation.class_id)))
            _, gt_annotation = gt_dict[name_gt]
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
    num_classes = len(gt_dataset.classes) + 1
    loss = DETRLoss(nc=num_classes, aux_loss=False, use_fl=False, use_vfl=False)
    batch = {
        'cls': gt_labels,
        'bboxes': gt_bboxes,
        'gt_groups': gt_groups
    }
    pred_bboxes = pred_bboxes.unsqueeze(0).unsqueeze(0)
    pred_scores = pred_scores.unsqueeze(0).unsqueeze(0)
    loss_output = loss.forward(
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        batch=batch
    )
    total_loss = 5*loss_output['loss_class'] + loss_output['loss_bbox']/1000 + loss_output['loss_giou']
    #record in wandb
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
            "prompt": prompt
        })
    return gt_class, TP, FP, FN, acc, F1, dataset, total_loss, prompt

def step(config: None, gt_dataset: sv.DetectionDataset, prompt: str, eval_metrics: bool = False):
    args = argparse.Namespace(
        config='/zhome/4a/b/137804/Desktop/autolbl/config.json',
        section='defects',
        model='Florence',
        tag='default',
        sahi=False,
        reload=False,
        ontology=f'{prompt}: defect',
        wandb=False,
        save_images = False
    )
    dataset = run_any_args(args)
    if eval_metrics:
        confusion_matrix, acc, map_result = evaluate_detections(dataset, gt_dataset)
        #compare_plot(dataset=dataset,gt_dataset=dataset)
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

def decode_prompt(tokenizer, input_ids):
    rounded_prompt = torch.round(input_ids)
    rounded_prompt = torch.clamp(rounded_prompt, 0, len(tokenizer)).to(torch.int64)
    prompt = tokenizer.decode(rounded_prompt, skip_special_tokens=True)
    return prompt

def main():
    torch.cuda.empty_cache()
    wandb.login()
    run = wandb.init(project="backprop")
    with open('config.json', 'r') as f:
        config = json.load(f)["defects"]
    gt_dataset = load_dataset(
        config['GT_IMAGES_DIRECTORY_PATH'],
        config['GT_ANNOTATIONS_DIRECTORY_PATH'],
        config['GT_DATA_YAML_PATH']
    )

    #defect only as class
    gt_dataset = set_one_class(gt_dataset=gt_dataset)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # Tokenize the prompt and create a tensor for optimization.
    input_ids = tokenizer("defect in wood sample")['input_ids']
    input_ids = torch.tensor(input_ids, dtype=torch.float32, requires_grad=True)

    # Wrap the loss and its gradient for BFGS.
    def objective(x_np):
        # Convert the flattened NumPy array back into a tensor with the original shape.
        x_tensor = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        # Compute the loss and get the prompt (for logging).
        gt_class, TP, FP, FN, acc, F1, dataset, loss, prompt = loss2(
            config=config,
            gt_dataset=gt_dataset,
            input_ids=x_tensor,
            tokenizer=tokenizer,
            eval_metrics=True  # Set to True if you want to compute metrics
        )
        print(f"Loss: {loss}, Prompt: {prompt}")
        return loss

    # Initial guess as a flattened vector.
    x0 = input_ids.detach().numpy().flatten()
    # Use SciPy's minimize
    #bounds = [(0, len(tokenizer))] * len(x0)
    #print(bounds)
    result = minimize(objective, x0, jac=False, options={'maxiter': 3})
    optimized_x = result.x
    optimized_tensor = torch.tensor(optimized_x, dtype=torch.float32)
    # For logging, round and clamp the optimized values before decoding.
    optimized_prompt = decode_prompt(tokenizer=tokenizer,input_ids=optimized_tensor)
    print("Optimized prompt:", optimized_prompt)
    wandb.log({
        "optimized_prompt": optimized_prompt,
        "final_loss": result.fun
    })
    run.finish()

if __name__ == "__main__":
    main()
