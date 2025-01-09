import wandb
import os
import re
import torch
import cv2
from utils.check_labels import *
from autodistill.detection import CaptionOntology
from autodistill_florence_2 import Florence2
from autodistill_grounding_dino import GroundingDINO
from utils.composed_detection_model import ComposedDetectionModel2
#from utils.config_work3 import *
from utils.config import *
tag = "Florence"
#tag = "DINO"
# Initialize wandb
wandb.login()
wandb.init(project="auto_label", name=f"Single {tag}", tags=f"{tag}")  # Updated project and run name
reset_folders(DATASET_DIR_PATH, "results")
sahi=False
# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())

# Display image sample
image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["bmp", "jpg", "jpeg", "png"]
)
print('Image count:', len(image_paths))

# Define ontology
with open("data/Semantic Map Specification.txt", "r") as file:
    content = file.read()
names = re.findall(r"name=([^\n]+)", content)
names = [name.lower().replace("_", " ") for name in names]
#ont_list = {(f"{name} surface defect"): name for name in names}
#ont_list = {(f"{name}"): name for name in names}
ont_list = {
    "Live knots are solid and cannot be knocked loose because they are fixed by growth or position in the wood structure. They are partially or completely intergrown within the growth rings.": "live knot",
    "Dead knots are loose knots that can fall out of the lumber when pushed or have already fallen out. They are caused by a dead branch that was not fully integrated into the tree before it was cut down.": "dead knot",
    "A knothole is a hole left where the knot has been knocked out.": "knot missing",
    "A knot with a crack indicates a knot that has developed a fissure, potentially compromising the wood's structural integrity.": "knot with crack",
    "Cracks, also known as checks, are ruptures or separations in the wood grain which reduce a board's appearance, strength, or utility.": "crack",
    "Quartzity refers to a specific type of wood defect characterized by the presence of mineral streaks or deposits within the wood.": "quartzity",
    "Resin pockets are accumulations of resin within the wood, often appearing as dark, sticky areas that can affect finishing processes.": "resin",
    "Marrow in wood refers to the presence of pith or soft tissue remnants, which can appear as a defect affecting the uniformity of the wood.": "marrow",
    "Blue stain is a discoloration caused by fungal infection, leading to blue or grayish streaks in the wood without affecting its structural integrity.": "blue stain",
    "Overgrown defects occur when a tree grows over a wound or foreign object, leading to irregular wood grain patterns and potential weaknesses.": "overgrown"
}
print(ont_list)# Convert mapping into a table
table = wandb.Table(columns=["prompt", "caption"])
for key, value in ont_list.items():
    table.add_data(key, value)

# Log the table
wandb.log({"Prompt Table": table})



convert_bmp_to_jpg(IMAGE_DIR_PATH)

# Initiate base model and autolabel
if tag == "DINO":
    base_model = GroundingDINO(ontology=CaptionOntology(ont_list))
if tag == "Florence":
    base_model = Florence2(ontology=CaptionOntology(ont_list))


# Log model settings
wandb.config.update({
    "model": (f"{base_model.__class__.__name__}"),
    "ontology": ont_list,
    "input_folder": IMAGE_DIR_PATH,
    "output_folder": DATASET_DIR_PATH,
    "sahi": sahi
})

# Label the dataset

dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=DATASET_DIR_PATH,
    sahi=sahi
)
#check if the dataset is empty
if len(dataset)==0:
    dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".jpg",
    output_folder=DATASET_DIR_PATH,
    sahi=sahi
    )

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH
)
print("Dataset size:", len(dataset))

# Call the function to plot annotated images
print(SAMPLE_SIZE)
#plot_annotated_images(dataset, SAMPLE_SIZE, "results/sample_annotated_images_grid.png")

# Evaluate the dataset
update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
compare_classes(gt_dataset, dataset)
compare_image_keys(gt_dataset, dataset)
evaluate_detections(dataset, gt_dataset)
# Log the size of the dataset
wandb.log({"dataset_size": len(dataset)})
# Save results images to wandb
plot_annotated_images(dataset, SAMPLE_SIZE, "results/sample_annotated_images_grid.png")

compare_plot(dataset, gt_dataset)



# Finish the wandb run
wandb.finish()