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
from autodistill.detection.detection_ontology import DetectionOntology

class Qwen25VLModel(DetectionBaseModel):
    """
    Detection model powered by Qwen 2.5 VL (Vision-Language) model.
    """
    
    def __init__(
        self,
        ontology: DetectionOntology,
        model_id_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        hf_token: Optional[str] = None,
        device: str = "auto",
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.NONE,
        prompt: str = "Outline the position of each defect and output all the coordinates in JSON format.",
        system_message: Optional[str] = None,
        max_new_tokens: int = 1024
    ):
        """
        Initialize the Qwen 2.5 VL model for detection.
        
        Args:
            ontology: The detection ontology to use
            model_id_or_path: Model ID or path to the model weights
            hf_token: Hugging Face token for downloading models
            device: Device to use for inference ("auto", "cpu", "cuda", etc.)
            optimization_strategy: Optimization strategy for the model
            prompt: Prompt to use for detection
            system_message: Optional system message to guide the model
            max_new_tokens: Maximum number of tokens to generate
        """
        super().__init__(ontology)
        
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            
        self.device = parse_device_spec(device)
        self.prompt = prompt
        self.system_message = system_message
        self.max_new_tokens = max_new_tokens
        
        # Load the model and processor
        self.processor, self.model = load_model(
            model_id_or_path=model_id_or_path,
            optimization_strategy=optimization_strategy
        )

    def predict(self, input_image: Union[str, np.ndarray, Image.Image]) -> sv.Detections:
        """
        Run detection on the input image.
        
        Args:
            input_image: The input image path, numpy array, or PIL Image

        Returns:
            sv.Detections: Detection results
        """
        # Convert input to PIL Image if necessary
        if isinstance(input_image, str):
            image = Image.open(input_image)
        elif isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image)
        else:
            image = input_image
        
        # Get original image dimensions
        resolution_wh = image.size
        
        # Run inference
        response, input_wh = self._run_inference(image)
        
        # Convert to Detections object
        detections = Detections.from_vlm(
            vlm="qwen_2_5_vl",
            result=response,
            input_wh=input_wh,
            resolution_wh=resolution_wh
        )
        
        return detections
    
    def _run_inference(self, image: Image.Image) -> Tuple[str, Tuple[int, int]]:
        """
        Run inference with the Qwen 2.5 VL model.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Tuple containing the model's response text and input dimensions
        """
        conversation = format_conversation(
            image=image, 
            prefix=self.prompt, 
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

        # Calculate input dimensions
        input_h = inputs['image_grid_thw'][0][1] * 14
        input_w = inputs['image_grid_thw'][0][2] * 14

        # Run prediction
        response = predict_with_inputs(
            **inputs,
            model=self.model,
            processor=self.processor,
            device=self.device,
            max_new_tokens=self.max_new_tokens
        )[0]

        return response, (input_w, input_h)