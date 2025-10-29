"""
Florence-2 Caption and Grounding Example Script

This script demonstrates how to use Florence-2 for:
1. Generating detailed captions
2. Caption-to-phrase grounding (detecting objects mentioned in captions)

This is an example/experimental script showing Florence-2 captioning capabilities.
For production use, see autolbl/models/florence.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import wandb

from autolbl.visualization.wandb import detections_to_wandb


# Configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MODEL_NAME = "microsoft/Florence-2-large"

# Initialize model and processor
print(f"Loading Florence-2 model on {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=TORCH_DTYPE, 
    trust_remote_code=True
).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model loaded successfully!")


def run_florence_task(task_prompt: str, image: Image.Image, text_input: str = None) -> dict:
    """
    Run a Florence-2 task on an image.
    
    Args:
        task_prompt: Florence task type (e.g., '<MORE_DETAILED_CAPTION>', '<CAPTION_TO_PHRASE_GROUNDING>')
        image: PIL Image
        text_input: Optional text input for the task
        
    Returns:
        Dictionary with task results
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
        
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed_answer


def caption_and_ground_images(
    input_folder: str,
    output_folder: str = None,
    extension: str = ".jpg",
    wandb_project: str = "Florence2",
    wandb_group: str = "caption_grounding"
) -> pd.DataFrame:
    """
    Generate captions and perform phrase grounding for images in a folder.
    
    This function:
    1. Generates detailed captions for each image
    2. Uses the caption to detect objects mentioned in it
    3. Logs results to W&B
    
    Args:
        input_folder: Path to input images
        output_folder: Path to save results (default: input_folder + "_labeled")
        extension: Image file extension
        wandb_project: W&B project name
        wandb_group: W&B group name
        
    Returns:
        DataFrame with results
    """
    wandb.init(project=wandb_project, group=wandb_group)
    
    if output_folder is None:
        output_folder = input_folder + "_labeled"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all images
    image_paths = list(Path(input_folder).glob(f"*{extension}"))
    
    if not image_paths:
        print(f"No images found with extension {extension} in {input_folder}")
        return pd.DataFrame()
    
    results_data = []
    
    progress_bar = tqdm(image_paths, desc="Processing images")
    for image_path in progress_bar:
        progress_bar.set_description(desc=f"Processing {image_path.name}", refresh=True)
        
        # Load image
        image = Image.open(image_path)
        
        # Step 1: Generate detailed caption
        caption_results = run_florence_task('<MORE_DETAILED_CAPTION>', image)
        caption_text = caption_results['<MORE_DETAILED_CAPTION>']
        
        # Step 2: Use caption for phrase grounding
        grounding_results = run_florence_task(
            '<CAPTION_TO_PHRASE_GROUNDING>',
            image,
            caption_text
        )
        
        # Extract detections
        detections = grounding_results['<CAPTION_TO_PHRASE_GROUNDING>']
        classes = detections['labels']
        bboxes = detections['bboxes']
        
        # Format detections for W&B
        formatted_detections = [
            [np.array(bbox), 1.0, 1.0, i] 
            for i, bbox in enumerate(bboxes)
        ]
        
        wandb_image = detections_to_wandb(image, formatted_detections, classes)
        
        # Store results
        results_data.append({
            'image_name': image_path.name,
            'caption': caption_text,
            'num_detections': len(bboxes),
            'detected_classes': ', '.join(classes),
            'wandb_image': wandb_image
        })
    
    # Create DataFrame and log to W&B
    df = pd.DataFrame(results_data)
    wandb.log({"results_table": wandb.Table(dataframe=df)})
    wandb.finish()
    
    print(f"\nProcessed {len(image_paths)} images")
    print(f"Results saved to: {output_folder}")
    
    return df


if __name__ == "__main__":
    # Example usage
    results_df = caption_and_ground_images(
        input_folder='images',
        output_folder='results',
        wandb_project='Florence2',
        wandb_group='wood_captioning'
    )
    
    print("\nSample Results:")
    print(results_df[['image_name', 'caption', 'num_detections']].head())
