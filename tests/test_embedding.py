# from autodistill_clip import CLIP
# from autodistill_metaclip import MetaCLIP
from utils.metaclip_model import MetaCLIP
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
#from autodistill.core import EmbeddingOntologyImage
from utils.embedding_ontology import EmbeddingOntologyImage
from utils.composed_detection_model import ComposedDetectionModel2
from autodistill.utils import plot
import torch
import open_clip
from PIL import Image
import os
import cv2

custom_composed = True
device = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_FOLDER = "Image_Embeddings"
DATASET_INPUT = "./images"
DATASET_OUTPUT = "./dataset"
PROMPT = "defect in wood e.g. knots, cracks, etc."
IMAGE = "./images/100000000.jpg"
images = os.listdir(INPUT_FOLDER)

# Create full paths for images
images_to_classes = {
    os.path.join(INPUT_FOLDER, "100000010.jpg"): "live knot",
    os.path.join(INPUT_FOLDER, "100100010.jpg"): "dead knot", 
    os.path.join(INPUT_FOLDER, "101800000.jpg"): "knot missing",
    os.path.join(INPUT_FOLDER, "100000082.jpg"): "knot with crack",
    os.path.join(INPUT_FOLDER, "100500053.jpg"): "crack",
    os.path.join(INPUT_FOLDER, "100000001.jpg"): "quartzity",
    os.path.join(INPUT_FOLDER, "101100021.jpg"): "resin",
    os.path.join(INPUT_FOLDER, "101900001.jpg"): "marrow",
    os.path.join(INPUT_FOLDER, "139100026.jpg"): "overgrown",
    os.path.join(INPUT_FOLDER, "144100014.jpg"): "blue stain"
}

# Verify images exist
for class_name, image_path in images_to_classes.items():
    if not os.path.exists(image_path):
        print(f"Warning: Image not found for class {class_name}: {image_path}")

HOME = os.path.expanduser("~")
img_emb = EmbeddingOntologyImage(images_to_classes)

class_model = MetaCLIP(img_emb)
if custom_composed:
    model = ComposedDetectionModel2(
        detection_model=GroundingDINO(CaptionOntology({PROMPT: PROMPT})),
        classification_model=class_model,
    )
else:
    model = class_model

result = model.predict(IMAGE)

plot(image=cv2.imread(IMAGE),
     classes=class_model.ontology.prompts(),
     detections=result)

print(result.class_id)
