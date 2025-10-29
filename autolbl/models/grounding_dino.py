import os
from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

torch.use_deterministic_algorithms(False)

import supervision as sv
from autodistill.detection import CaptionOntology#, DetectionBaseModel
from autolbl.models.detection_base import DetectionBaseModel
from autodistill.helpers import load_image
from groundingdino.util.inference import Model

from autodistill_grounding_dino.helpers import (combine_detections,
                                                load_grounding_dino)
from autolbl.ontology.embedding import EmbeddingOntologyImage

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GroundingDINO(DetectionBaseModel):
    ontology: CaptionOntology
    grounding_dino_model: Model
    box_threshold: float
    text_threshold: float

    def __init__(
        self, ontology: CaptionOntology, box_threshold=0.35, text_threshold=0.25
    ):
        self.ontology = ontology
        self.grounding_dino_model = load_grounding_dino()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        if isinstance(self.ontology, EmbeddingOntologyImage):
            self.ontology.process(self)

    def predict(self, input: str) -> sv.Detections:
        image = load_image(input, return_format="cv2")

        detections_list = []
        if isinstance(self.ontology, EmbeddingOntologyImage):

            for _, description in enumerate(self.ontology.classes()):
                detections = self.grounding_dino_model.predict_with_caption(
                    image=image,
                    classes = [description],
                    box_threshold= self.box_threshold,
                    text_threshold=self.text_threshold
                )
                detections_list.append(detections)
        else:
            for _, description in enumerate(self.ontology.prompts()):
                detections = self.grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=[description],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )

                detections_list.append(detections)

        detections = combine_detections(
            detections_list, overwrite_class_ids=range(len(detections_list))
        )

        return detections
