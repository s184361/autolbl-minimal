import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from PIL import Image
from autolbl.ontology.embedding import EmbeddingOntologyImage
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotImageClassification,
    AutoTokenizer,
)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class MetaCLIP(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):

        self.preprocess = AutoProcessor.from_pretrained("facebook/metaclip-b16-fullcc2.5b")
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            "facebook/metaclip-b16-fullcc2.5b", torch_dtype=torch.float16
        ).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-b16-fullcc2.5b")
        self.model = torch.compile(model)
        self.ontology = ontology
        if isinstance(ontology, EmbeddingOntologyImage):
            self.ontology.process(self)

    def predict(self, input: str, confidence: int = 0.5) -> sv.Classifications:
        prompts = self.ontology.prompts()
        image = Image.open(input)
        # check if the ontology is a EmbeddingOntologyImage
        with torch.no_grad():
            # cosine similarity as logits
            image_features = self.embed_image(image)
            if isinstance(self.ontology, EmbeddingOntologyImage):
                probs = []
                embeddings = self.ontology.classes()
                for embedding in embeddings:
                    probs.append(self.compare(image_features, embedding[0]))
                probs = [probs]
            else:
                text_inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                text_features = self.model.get_text_features(**text_inputs)

                probs = (image_features @ text_features.T).softmax(dim=-1).cpu()
        # create dictionary of prompt: probability
        probs = list(zip(prompts, probs[0]))

        # filter out prompts with confidence less than the threshold
        probs = [i for i in probs if i[1] > confidence]

        return sv.Classifications(
            class_id=np.array([prompts.index(i[0]) for i in probs]),
            confidence=np.array([i[1] for i in probs]),
        )

    def embed_image(self, input: str) -> torch.Tensor:

        inputs = self.preprocess(images=input, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast('cuda'): 
                outputs = self.model.get_image_features(**inputs)

            return outputs

    def embed_text(self, input: str) -> torch.Tensor:
        inputs = self.processor(text=input, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            return outputs

    def compare(self, embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        return torch.cosine_similarity(embed1, embed2).item()
