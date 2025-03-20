from utils.qwen25_model import Qwen25VL
from autodistill.detection import CaptionOntology
#from maestro.trainer.models.qwen_2_5_vl.checkpoints import load_model, OptimizationStrategy
from utils.check_labels import evaluate_detections
# Define your ontology
ontology = CaptionOntology({"defect":"defect"})

# Initialize the model
model = Qwen25VL(
    ontology=ontology,
    hf_token="os.getenv("HF_TOKEN", "")",
)

# Run inference on a single image
detections = model.predict("/zhome/4a/b/137804/Desktop/autolbl/data/bottle/images/000_broken_large.jpg")
print(detections)
# Or label a folder of images
dataset = model.label(
    input_folder="/zhome/4a/b/137804/Desktop/autolbl/data/bottle/images/",
    output_folder="results",
    save_images=False,
)

