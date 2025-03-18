#%%
from maestro.trainer.models.qwen_2_5_vl.inference import predict
from maestro.trainer.models.qwen_2_5_vl.checkpoints import load_model, OptimizationStrategy
import os
import PIL
import json
from PIL import Image
from typing import Optional, Tuple, Union

from maestro.trainer.models.qwen_2_5_vl.inference import predict_with_inputs
from maestro.trainer.models.qwen_2_5_vl.loaders import format_conversation
from maestro.trainer.common.utils.device import parse_device_spec
from qwen_vl_utils import process_vision_info
os.environ["HF_TOKEN"] = "os.getenv("HF_TOKEN", "")"

# Use the correct model ID
processor, model = load_model(
    model_id_or_path="Qwen/Qwen2.5-VL-7B-Instruct",  # Corrected model ID
    optimization_strategy=OptimizationStrategy.NONE
)
with open("/zhome/4a/b/137804/Desktop/autolbl/config.json", 'r') as f:
    config = json.load(f)["wood"]
image = PIL.Image.open(config["IMAGE_DIR_PATH"]+"/000_combined.jpg")

def run_qwen_2_5_vl_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    system_message: Optional[str] = None,
    device: str = "auto",
    max_new_tokens: int = 1024,
) -> Tuple[str, Tuple[int, int]]:
    device = parse_device_spec(device)
    conversation = format_conversation(image=image, prefix=prompt, system_message=system_message)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversation)

    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt",
    )

    input_h = inputs['image_grid_thw'][0][1] * 14
    input_w = inputs['image_grid_thw'][0][2] * 14

    response = predict_with_inputs(
        **inputs,
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=max_new_tokens
    )[0]

    return response, (input_w, input_h)


SYSTEM_MESSAGE = None
PROMPT = "Outline the position of each defect and output all the coordinates in JSON format."

resolution_wh = image.size
response, input_wh = run_qwen_2_5_vl_inference(
    model=model,
    processor=processor,
    image=image,
    prompt=PROMPT,
    system_message=SYSTEM_MESSAGE
)

print(response)

from utils.core import Detections
import supervision as sv
detections = Detections.from_vlm(
    vlm="qwen_2_5_vl",
    result=response,
    input_wh=input_wh,
    resolution_wh=resolution_wh
)

box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

annotated_image = image.copy()
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
#save annotated image
annotated_image.save("annotated.jpg")