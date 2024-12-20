import os
import re
import torch
import cv2
import supervision as sv
import matplotlib.pyplot as plt
from utils.check_labels import *
from autodistill.detection import CaptionOntology
#from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill_florence_2 import Florence2
from utils.composed_detection_model import ComposedDetectionModel2
from utils.embedding_ontology import EmbeddingOntologyImage
from utils.metaclip_model import MetaCLIP
from utils.config import *

def main():
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    DATASET_DIR_PATH = f"{HOME}/dataset"
    reset_folders(DATASET_DIR_PATH, "results")

    # Display image sample
    image_paths = sv.list_files_with_extensions(
        directory=IMAGE_DIR_PATH,
        extensions=["bmp", "jpg", "jpeg", "png"]
    )
    print('Image count:', len(image_paths))

    titles = [image_path.stem for image_path in image_paths[:SAMPLE_SIZE]]
    images = [cv2.imread(str(image_path)) for image_path in image_paths[:SAMPLE_SIZE]]
    plt.ion()
    sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
    plt.savefig("results/sample_images_grid.png")

    # Define ontology
    with open("data/Semantic Map Specification.txt", "r") as file:
        content = file.read()
    names = re.findall(r"name=([^\n]+)", content)
    names = [name.lower().replace("_", " ") for name in names]
    # Embeding dir
    EMBEDING_DIR_PATH = os.path.abspath(os.path.join(HOME, "Image_Embeddings"))
    convert_bmp_to_jpg(EMBEDING_DIR_PATH)

    # Update paths to use os.path.join for proper path formatting
    """
    images_to_classes = {
        os.path.join(EMBEDING_DIR_PATH, "100000010.jpg"): "live knot",
        os.path.join(EMBEDING_DIR_PATH, "100100010.jpg"): "dead knot",
        os.path.join(EMBEDING_DIR_PATH, "101800000.jpg"): "knot missing", 
        os.path.join(EMBEDING_DIR_PATH, "100000082.jpg"): "knot with crack",
        os.path.join(EMBEDING_DIR_PATH, "100500053.jpg"): "crack",
        os.path.join(EMBEDING_DIR_PATH, "100000001.jpg"): "quartzity",
        os.path.join(EMBEDING_DIR_PATH, "101100021.jpg"): "resin",
        os.path.join(EMBEDING_DIR_PATH, "101900001.jpg"): "marrow",
        os.path.join(EMBEDING_DIR_PATH, "139100026.jpg"): "overgrown",
        os.path.join(EMBEDING_DIR_PATH, "144100014.jpg"): "blue stain"
    }
    """
    images_to_classes = {
        os.path.join(f"{HOME}/croped_images", "live knot.jpg"): "live knot",
        os.path.join(f"{HOME}/croped_images", "dead knot.jpg"): "dead knot",
        os.path.join(f"{HOME}/croped_images", "knot missing.jpg"): "knot missing",
        os.path.join(f"{HOME}/croped_images", "knot with crack.jpg"): "knot with crack",
        os.path.join(f"{HOME}/croped_images", "crack.jpg"): "crack",
        os.path.join(f"{HOME}/croped_images", "quartzity.jpg"): "quartzity",
        os.path.join(f"{HOME}/croped_images", "resin.jpg"): "resin",
        os.path.join(f"{HOME}/croped_images", "marrow.jpg"): "marrow",
        os.path.join(f"{HOME}/croped_images", "overgrown.jpg"): "overgrown",
        os.path.join(f"{HOME}/croped_images", "blue stain.jpg"): "blue stain",
    }
    
    # Verify images exist
    for image_path, class_name in images_to_classes.items():
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for class {class_name}: {image_path}")

    # Create embedding ontology and models
    img_emb = EmbeddingOntologyImage(images_to_classes)
    class_model = MetaCLIP(img_emb)

    # Initiate base model and autolabel
    DATASET_DIR_PATH = f"{HOME}/dataset"

    convert_bmp_to_jpg(IMAGE_DIR_PATH)

    # Create a combined model that uses both GroundingDINO for detection and MetaCLIP for classification
    model = ComposedDetectionModel2(
        detection_model=Florence2(CaptionOntology({"defect": "defect"})),
        classification_model=class_model,
    )

    # Label dataset
    dataset = model.label(
        input_folder=IMAGE_DIR_PATH, extension=".png", output_folder=DATASET_DIR_PATH
    )

    # Load and evaluate results
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=IMAGES_DIRECTORY_PATH,
        annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
        data_yaml_path=DATA_YAML_PATH
    )
    print("Dataset size:", len(dataset))

    image_names = list(dataset.images.keys())[:SAMPLE_SIZE]

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()

    images = []
    for image_name in image_names:
        image = dataset.images[image_name]
        annotations = dataset.annotations[image_name]
        labels = [dataset.classes[class_id] for class_id in annotations.class_id]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=annotations)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations)
        images.append(annotated_image)

    plt.ion()
    sv.plot_images_grid(
        images=images, 
        titles=image_names, 
        grid_size=SAMPLE_GRID_SIZE, 
        size=SAMPLE_PLOT_SIZE
    )
    plt.savefig("results/sample_annotated_images_grid.png", dpi=300)

    sv.plot_images_grid(images=images, titles=image_names, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)
    # save in high resolution
    plt.savefig("results/sample_annotated_images_grid.png", dpi=1200)

    # evaluate the dataset
    update_labels(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
    gt_dataset = load_dataset(GT_IMAGES_DIRECTORY_PATH, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH)
    compare_classes(gt_dataset, dataset)
    compare_image_keys(gt_dataset, dataset)
    evaluate_detections(dataset, gt_dataset)
    compare_plot(dataset,gt_dataset)
if __name__ == "__main__":
    main()
