# from autodistill_clip import CLIP
from autodistill_metaclip import MetaCLIP
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from autodistill.core import EmbeddingOntologyImage
from autodistill.core.composed_detection_model import ComposedDetectionModel
from composed_detection_model import ComposedDetectionModel2
from autodistill.utils import plot
from autodistill.core.embedding_model import EmbeddingModel
import torch
import open_clip
from PIL import Image
import os
import cv2

custom_composed = True
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

INPUT_FOLDER = "Image_Embeddings"
DATASET_INPUT = "./images"
DATASET_OUTPUT = "./dataset"
PROMPT = "defect in wood e.g. knots, cracks, etc."
IMAGE = "./images/100000000.jpg"
images = os.listdir(INPUT_FOLDER)


# Create full paths for images
images_to_classes = {
    "live knot": os.path.join(INPUT_FOLDER, "100000010.jpg"),
    "dead knot": os.path.join(INPUT_FOLDER, "100100010.jpg"),
    "knot missing": os.path.join(INPUT_FOLDER, "101800000.jpg"),
    "knot with crack": os.path.join(INPUT_FOLDER, "100000082.jpg"),
    "crack": os.path.join(INPUT_FOLDER, "100500053.jpg"),
    "quartzity": os.path.join(INPUT_FOLDER, "100000001.jpg"),
    "resin": os.path.join(INPUT_FOLDER, "101100021.jpg"),
    "marrow": os.path.join(INPUT_FOLDER, "101900001.jpg"),
    "overgrown": os.path.join(INPUT_FOLDER, "139100026.jpg"),
    "blue stain": os.path.join(INPUT_FOLDER, "144100014.jpg"),
}

# Verify images exist
for class_name, image_path in images_to_classes.items():
    if not os.path.exists(image_path):
        print(f"Warning: Image not found for class {class_name}: {image_path}")

HOME = os.path.expanduser("~")
img_emb = EmbeddingOntologyImage(images_to_classes)


model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
 )

class_model = MetaCLIP(img_emb)
img_emb.process(class_model)
print(img_emb.embeddingMap)

if custom_composed:
    model = ComposedDetectionModel2(
        detection_model=GroundingDINO(CaptionOntology({PROMPT: PROMPT})),
        classification_model=class_model,
    )
else:
    model = ComposedDetectionModel(
        detection_model=GroundingDINO(CaptionOntology({PROMPT: PROMPT})),
        classification_model=class_model,
    )

result = model.predict(IMAGE)

print(class_model.ontology.classes())

plot(image=cv2.imread(IMAGE),
     classes=class_model.ontology.classes(),
     detections=result)
