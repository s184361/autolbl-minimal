import os
import PIL
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union

from maestro.trainer.models.qwen_2_5_vl.inference import predict_with_inputs
from maestro.trainer.models.qwen_2_5_vl.checkpoints import load_model, OptimizationStrategy
from maestro.trainer.models.qwen_2_5_vl.loaders import format_conversation
from maestro.trainer.common.utils.device import parse_device_spec
from qwen_vl_utils import process_vision_info
import supervision as sv

from utils.core import Detections
from utils.detection_base_model import DetectionBaseModel
from autodistill.helpers import load_image
from autodistill.detection import (CaptionOntology,
                                   DetectionTargetModel)
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
class Qwen25VL(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self,ontology: CaptionOntology, hf_token: Optional[str] = None):
        # Call the parent class's __init__ method
        super().__init__(ontology)
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        model_id_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        revision = "refs/heads/main"
        trust_remote_code = True
        cache_dir = None
        device = "auto"
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        self.device = parse_device_spec(device)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            revision="refs/heads/main",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=None,
        )
        self.model.to(self.device)
        
        self.max_new_tokens = 1024
        self.system_message = None
    
    def predict(self, input: str, confidence: float = 0.5) -> sv.Detections:
        image = load_image(input, return_format="PIL")
        ontology_prompts = self.ontology.prompts()

        PROMPT = "Outline the position of the objects according to description '" + ", ".join(ontology_prompts) + ".' Output all the coordinates in JSON format."
        #print(PROMPT)
        resolution_wh = image.size
        response, input_wh = self.run_qwen_2_5_vl_inference(
            image=image,
            prompt=PROMPT
        )
        
        detections = Detections.from_vlm(
            vlm="qwen_2_5_vl",
            result=response,
            input_wh=input_wh,
            resolution_wh=resolution_wh
        )

        if confidence > 0 and hasattr(detections, 'confidence') and detections.confidence is not None:
            detections = detections[detections.confidence > confidence]
        #print("last detection")
        #print(detections)
        return detections

    def run_qwen_2_5_vl_inference(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Tuple[str, Tuple[int, int]]:
        conversation = format_conversation(
            image=image, 
            prefix=prompt, 
            system_message=self.system_message
        )
        
        text = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, _ = process_vision_info(conversation)

        inputs = self.processor(
            text=text,
            images=image_inputs,
            return_tensors="pt",
        )

        input_h = inputs['image_grid_thw'][0][1] * 14
        input_w = inputs['image_grid_thw'][0][2] * 14

        # Don't pass the device parameter here - Accelerate handles it
        response = predict_with_inputs(
            **inputs,
            model=self.model,
            processor=self.processor,
            max_new_tokens=self.max_new_tokens,
            device=self.device
        )[0]

        return response, (input_w, input_h)