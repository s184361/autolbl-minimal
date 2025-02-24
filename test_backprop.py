import argparse
import json
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
from scipy.optimize import BFGS
def compute_loss(dataset = sv.DetectionDataset, gt_dataset = sv.DetectionDataset):

    pred_bboxes = torch.empty(0, 4)
    pred_labels = torch.empty(0)
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.empty(0)
    gt_dict = {os.path.splitext(os.path.basename(image_path))[0] + ".jpg": (image_path, annotation) for image_path, _, annotation in gt_dataset}

    for image_path, _, annotation in dataset:

        name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        if name_gt in gt_dict:
            pred_bboxes = torch.cat((pred_bboxes, torch.tensor(annotation.xyxy)))
            pred_labels = torch.cat((pred_labels, torch.tensor(annotation.class_id)))
            _, gt_annotation = gt_dict[name_gt]
            gt_bboxes = torch.cat((gt_bboxes, torch.tensor(gt_annotation.xyxy)))
            gt_labels = torch.cat((gt_labels, torch.tensor(gt_annotation.class_id)))

    #one hot encode labels
    pred_labels = F.one_hot(pred_labels.long()).float()
    gt_labels = F.one_hot(gt_labels.long()).float()

    # Step 2: Classification Loss
    classification_loss = F.cross_entropy(pred_labels, gt_labels)

    # Step 3: Bounding Box Loss
    bbox_l1_loss = F.l1_loss(pred_bboxes, gt_bboxes)/gt_bboxes.shape[0]
    bbox_giou_loss = 1 - torch.diag(ops.generalized_box_iou(pred_bboxes, gt_bboxes)).mean()
    # Step 4: Total Loss
    total_loss = classification_loss + 5 * bbox_l1_loss + 2 * bbox_giou_loss
    return total_loss

def loss2(config: None, gt_dataset: sv.DetectionDataset, input_ids: torch.Tensor, tokenizer: BertTokenizerFast, eval_metrics: bool = False):
    prompt = decode_prompt(tokenizer, input_ids)
    gt_class, TP, FP, FN, acc, F1, dataset = step(config, gt_dataset, prompt, eval_metrics)
    pred_bboxes = torch.empty(0, 4)
    pred_labels = torch.empty(0)
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.empty(0)
    gt_dict = {os.path.splitext(os.path.basename(image_path))[0] + ".jpg": (image_path, annotation) for image_path, _, annotation in gt_dataset}
    pred_scores = torch.empty(0, 2)  # Ensure pred_scores has the correct shape
    gt_groups = []
    # Collect data
    for image_path, _, annotation in dataset:
        name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        if name_gt in gt_dict:
            pred_bboxes = torch.cat((pred_bboxes, torch.tensor(annotation.xyxy)))
            pred_labels = torch.cat((pred_labels, torch.tensor(annotation.class_id)))
            _, gt_annotation = gt_dict[name_gt]
            #gt_groups (List[int]): List of length equal to batch size, containing the number of ground truths for each image.
            #gt_groups = gt_groups + [len(gt_annotation.xyxy)]

            gt_bboxes = torch.cat((gt_bboxes, torch.tensor(gt_annotation.xyxy)))
            gt_labels = torch.cat((gt_labels, torch.tensor(gt_annotation.class_id)))
            if annotation.confidence is not None:
                # Convert to tensor and ensure it's 1D
                conf_tensor = torch.tensor(annotation.confidence, dtype=torch.float32).flatten()  # shape [n]
                # Reshape to [n, 1] and stack to get [n, 2]
                conf_tensor = conf_tensor.unsqueeze(1)                        # shape [n, 1]
                score_tensor = torch.cat([1 - conf_tensor, conf_tensor], dim=1)  # shape [n, 2]
                pred_scores = torch.cat((pred_scores, score_tensor), dim=0)

    if len(pred_scores) == 0:
        pred_scores = torch.tensor([[-10, 10]])
        pred_scores = pred_scores.repeat(len(pred_labels), 1).float()
    if len(gt_groups) == 0:
        #list of 1 with the length of the batch size
        #gt_groups= torch.ones(len(gt_labels),dtype=int).tolist()
        gt_groups = [len(gt_labels)]
    # Process labels
    pred_labels = F.one_hot(pred_labels.long()).float()
    gt_labels = gt_labels.to(torch.int64)
    num_classes = len(gt_dataset.classes)+1


    loss = DETRLoss(nc=num_classes, aux_loss=False, use_fl=False, use_vfl=False)
    batch = {
        'cls': gt_labels,  # [num_gts]
        'bboxes': gt_bboxes,  # [num_gts,4]
        'gt_groups': gt_groups  # [batch_size]
    }
    pred_bboxes=pred_bboxes.unsqueeze(0).unsqueeze(0)  # [1, 1, num_queries, 4]
    pred_scores=pred_scores.unsqueeze(0).unsqueeze(0)  # [1, 1, num_queries, num_classes]
    print(gt_groups, "[batch_size]")
    print(pred_bboxes.shape, "[l,b,query,4]")
    print(pred_scores.shape, "[l,b,query,num_classes]")

    loss_output = loss.forward(
        pred_bboxes=pred_bboxes,  # [1, 1, num_queries, 4]
        pred_scores=pred_scores,  # [1, 1, num_queries, num_classes]
        batch=batch
    )
    total_loss = loss_output['loss_class']+loss_output['loss_bbox']+loss_output['loss_giou']
    return gt_class, TP, FP, FN, acc, F1, dataset, total_loss, prompt

def step(config: None, gt_dataset: sv.DetectionDataset, prompt: str, eval_metrics: bool = False):
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
    dataset = run_any_args(args)

    if eval_metrics:
        confusion_matrix, acc, map_result=evaluate_detections(dataset, gt_dataset)
        acc = acc[0]
        #compare_plot(dataset, gt_dataset)
        #extract true positives
        print(f"Accuracy: {acc}")
        # return "class", "TP", "FP", "FN", "Accuracy", "F1"
        gt_class = "defect"
        TP = confusion_matrix[0, 0]/confusion_matrix.sum()
        FP = confusion_matrix[0, 1]/confusion_matrix.sum()
        FN = confusion_matrix[1, 0]/confusion_matrix.sum()
        F1 = 2*TP/(2*TP+FP+FN)
    else:
        gt_class = "defect"
        TP = None
        FP = None
        FN = None
        acc = None
        F1 = None
    
    return gt_class, TP, FP, FN, acc, F1, dataset

def decode_prompt(tokenizer, input_ids):
    #round to the nearest integer
    rounded_prompt = torch.round(input_ids)
    #check if the prompt is in the range of the tokenizer and set to 0 if it is not
    rounded_prompt = torch.clamp(rounded_prompt, 0, len(tokenizer)).to(torch.int64)

    prompt = tokenizer.decode(rounded_prompt, skip_special_tokens=True)
    return prompt

def main():
    torch.cuda.empty_cache()
    wandb.login()
    run = wandb.init(project="backprop")

    # Load configurations and datasets
    with open('config.json', 'r') as f:
        config = json.load(f)["defects"]
    gt_dataset = load_dataset(
        config['GT_IMAGES_DIRECTORY_PATH'],
        config['GT_ANNOTATIONS_DIRECTORY_PATH'],
        config['GT_DATA_YAML_PATH']
    )

    # Initialize the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # Tokenize the prompt; make the input tensor require gradients
    input_ids = tokenizer("blue stain crack dead knot knot missing knot with crack live knot marrow overgrown quartzity resin")['input_ids']
    input_ids = torch.tensor(input_ids, dtype=torch.float32, requires_grad=True)
    # Set up optimizer on the tokenized input
    input_optimizer = torch.optim.LBFGS([input_ids], lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(input_optimizer, T_max=10, eta_min=0)

    epochs = 10
    # Training loop
    for epoch in range(1, epochs + 1):
        input_optimizer.zero_grad()
        gt_class, TP, FP, FN, acc, F1, dataset, current_loss, prompt = loss2(config=config,
                                                                            gt_dataset=gt_dataset,
                                                                            input_ids=input_ids,
                                                                            tokenizer=tokenizer,
                                                                            eval_metrics=True)

        input_ids.grad, = torch.autograd.grad(current_loss, [input_ids])
        
        input_optimizer.step()
        input_optimizer.zero_grad()
        #current_loss.backward(retain_graph=True)
        input_optimizer.step()
        wandb.log({
                "epoch": epoch,
                "loss": current_loss,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "accuracy": acc,
                "F1": F1,
                "prompt": prompt
            })
        print(f"Epoch {epoch}: {gt_class}, {TP}, {FP}, {FN}, {acc}, {F1}")
    run.finish()
if __name__ == "__main__":
    main()
