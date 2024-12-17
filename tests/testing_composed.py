from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv
from config import *
#from composed_detection_model import ComposedDetectionModel
from composed_detection_model import ComposedDetectionModel2
#from autodistill.core.composed_detection_model import ComposedDetectionModel
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
classes = ["live knot", "Burger King"]


SAMCLIP = ComposedDetectionModel2(
    detection_model=GroundedSAM(
        CaptionOntology({"defect": "defect in wood e.g. knots, cracks, etc."})
    ),
    classification_model=CLIP(
        CaptionOntology({k: k for k in classes})
    )
)

IMAGE = "100000010.jpg"

results = SAMCLIP.predict(IMAGE)

image = cv2.imread(IMAGE)

annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{classes[int(class_id)]} {conf:.2f}"
    for class_id, conf in zip(results.class_id, results.confidence)
]

annotated_frame = annotator.annotate(
    scene=image.copy(), detections=results
)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame, labels=labels, detections=results
)

sv.plot_image(annotated_frame, size=(8, 8))

# Label entire dataset
dataset = SAMCLIP.label(
    input_folder=IMAGE_DIR_PATH, 
    extension=".jpg",  # Make sure this matches your image extensions
    output_folder=DATASET_DIR_PATH
)
