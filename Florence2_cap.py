#%%
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
import wandb  # Import wandb
from utils.wandb_utils import detections_to_wandb  # Import the utility function

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

#%%
def run_example(task_prompt, text_input=None,image=None):

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
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

def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.show()  
# %%
def label(
        #self,
        input_folder: str,
        extension: str = ".jpg",
        output_folder: str | None = None,
        human_in_the_loop: bool = False,
        roboflow_project: str | None = None,
        roboflow_tags: list[str] = ["autodistill"],
        sahi: bool = False,
        record_confidence: bool = False,
        #nms_settings: NmsSetting = NmsSetting.NONE,
        save_images=True
    ): #-> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        wandb.init(project="Florence2")
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        image_paths = glob.glob(input_folder + "/*" + extension)
        detections_map = {}

        progress_bar = tqdm(image_paths, desc="Labeling images")
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)

            image = Image.open(f_path)
            
            #task_prompt = '<MORE_DETAILED_CAPTION>'
            #results = run_example(task_prompt=task_prompt, image=image)
            text_input = "blue stain: blue stain, crack: crack, Dead knot or partly dead knot with a ring of bark around the circumference: dead knot, fallen out or partially fallen out knot: knot missing, cracks inside and around knot: knot with crack, fresh and firm knots or sound knot: live knot, marrow: marrow, overgrown: overgrown, quartzity: quartzity, resin pocket that is completly dry i.e no sticky resin completly crystalizd or resind pocket where resin is partly crstalized or complately in liquid form: resin, Firm black knot that is fixed with surrouding wood: black knot"#results[task_prompt]
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            results = run_example(task_prompt, text_input, image=image)
            results['<MORE_DETAILED_CAPTION>'] = text_input

            fig=plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])

            #save the image in results folder
            try:
                plt.savefig(output_folder + '/' + f_path)
            except:
                #create the folder
                os.makedirs(output_folder, exist_ok=True)
                plt.savefig(output_folder + '/' + f_path)
            plt.close()

            # Prepare data for WandB
            detections = results['<CAPTION_TO_PHRASE_GROUNDING>']
            classes = detections['labels']  # Assuming 'labels' are the class names
            bboxes = detections['bboxes']
            formatted_detections = []
            for i, bbox in enumerate(bboxes):
                formatted_detections.append([bbox, 1.0, i])  # Assuming confidence is 1.0 and class ID is the index

            wandb_image = detections_to_wandb(image, detections, classes)
            image_name = os.path.basename(f_path)
            wandb.log({image_name: wandb_image})
        wandb.finish()

label(input_folder='images', output_folder='results')