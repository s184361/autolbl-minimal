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

INPUT_FOLDER = "./samples"
DATASET_INPUT = "./images"
DATASET_OUTPUT = "./dataset"
PROMPT = "album cover"
IMAGE = "./images_test/example.jpeg"
images = os.listdir(INPUT_FOLDER)

# Create full paths for images
images_to_classes = {
    os.path.join(INPUT_FOLDER, "IMG_9022.jpeg"): "midnights",
    os.path.join(INPUT_FOLDER, "323601467684.jpeg"): "men amongst mountains",
    os.path.join(INPUT_FOLDER, "IMG_9056.jpeg"): "we are",
    os.path.join(INPUT_FOLDER, "Images (5).jpeg"): "oh wonder",
    os.path.join(INPUT_FOLDER, "Images (4).jpeg"): "brightside",
    os.path.join(INPUT_FOLDER, "Images (3).jpeg"): "tears for fears",
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
