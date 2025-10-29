# AutoLbl: Automatic Labelling for Image Datasets

## Introduction

AutoLbl is a framework for automatic image annotation using vision-language models (VLMs). Its primary purpose is to facilitate efficient defect detection and classification on industrial inspection datasets. AutoLbl uses state-of-the-art models like Florence-2, Grounding DINO, and Qwen for zero-shot object detection and automatic annotation generation. AutoLbl is currently developed and maintained by Jędrzej Kolbert at DTU (Danmarks Tekniske Universitet) as part of an MSc thesis project.

Initially, AutoLbl was designed specifically for wood defect detection and quality control in industrial settings, particularly enabling automated labeling of defects like knots, cracks, resin pockets, and color variations. The system integrates multiple vision models and provides flexible ontology-based detection, allowing users to define custom defect categories through natural language descriptions. However, AutoLbl's detection system and model composition are general, which means they can be easily adapted for use in other domains. The framework has been successfully tested on multiple datasets including MVTec AD (wood, bottle, tire categories) and the Zenodo wood defect dataset.

## Features

- **Multiple Vision Models**: Support for Florence-2, Grounding DINO, Qwen2.5-VL, and MetaCLIP
- **Model Composition**: Combine detection and classification models for improved accuracy
- **Flexible Ontology System**: Define defect classes using natural language descriptions
- **Automated Dataset Preparation**: Convert ground truth masks and annotations to YOLO format
- **Comprehensive Evaluation**: Built-in metrics including confusion matrices, precision, recall, F1, and mAP
- **W&B Integration**: Automatic experiment tracking and visualization with Weights & Biases
- **Multiple Dataset Support**: Pre-configured for MVTec AD and Zenodo wood datasets
- **NMS Options**: Support for class-agnostic and class-specific non-maximum suppression
- **CLI Tools**: Command-line interface for dataset preparation and inference

## Installation

### Install AutoLbl using pip
```bash
# Clone the repository
git clone https://github.com/s184361/autolbl.git
cd autolbl

# Install the package
pip install -e .
```

### Install for development
```bash
git clone https://github.com/s184361/autolbl.git
cd autolbl
pip install -e .[dev]
```

### Requirements
- Python >= 3.11, < 3.13
- CUDA-compatible GPU (recommended for inference)
- PyTorch with CUDA support
- See `pyproject.toml` for complete dependency list

## Package Structure

```
autolbl/
├── autolbl/                    # Main package
│   ├── models/                 # VLM model implementations
│   │   ├── florence.py         # Florence-2 model
│   │   ├── grounding_dino.py   # Grounding DINO model
│   │   ├── qwen.py             # Qwen2.5-VL model
│   │   ├── metaclip.py         # MetaCLIP classifier
│   │   └── composed.py         # Model composition utilities
│   ├── datasets/               # Dataset processing utilities
│   │   ├── dataset_prep.py     # Dataset preparation functions
│   │   ├── converters.py       # Format converters
│   │   └── processing.py       # Data processing utilities
│   ├── evaluation/             # Evaluation metrics
│   │   └── metrics.py          # Confusion matrix, mAP, F1, etc.
│   ├── ontology/               # Ontology management
│   │   └── embedding.py        # Embedding-based ontology
│   ├── visualization/          # Visualization tools
│   │   └── wandb.py            # W&B integration
│   ├── cli/                    # Command-line interface
│   │   ├── prepare.py          # Dataset preparation CLI
│   │   └── infer.py            # Inference CLI
│   └── core.py                 # Core utilities
├── examples/                   # Usage examples
├── experiments/                # Experimental scripts
│   ├── prompt_optimization/    # DSPy prompt optimization
│   └── embedding/              # Embedding experiments
├── config.json                 # Configuration file
├── pyproject.toml              # Package configuration
└── setup.py                    # Setup script
```

## Quick Start

### 1. Prepare Your Dataset

AutoLbl includes a CLI tool for dataset preparation:

```bash
# Prepare MVTec AD Wood dataset
autolbl-prepare --dataset wood

# Prepare Zenodo Images1 dataset (first 100 images)
autolbl-prepare --dataset images1

# Prepare all datasets
autolbl-prepare --dataset all
```

See [DATASET_PREPARATION.md](DATASET_PREPARATION.md) for detailed instructions.

### 2. Configure Your Experiment

Edit `config.json` to set up paths and parameters:

```json
{
  "local_wood": {
    "HOME": "/path/to/autolbl",
    "IMAGE_DIR_PATH": "/path/to/images",
    "GT_ANNOTATIONS_DIRECTORY_PATH": "/path/to/annotations",
    "PROMPT": "wood defect detection including knots, cracks, holes, and color variations"
  }
}
```

### 3. Run Detection

Use the CLI tool for inference:

```bash
# Basic usage with Florence-2
autolbl-infer --section local_wood --model Florence

# With custom ontology
autolbl-infer --section local_wood --model Florence \
  --ontology "color: color, combined: combined, hole: hole, liquid: liquid, scratch: scratch"

# With class-specific NMS
autolbl-infer --section local_wood --model Florence \
  --ontology "color: color, hole: hole, scratch: scratch" \
  --nms class_specific

# Using Grounding DINO
autolbl-infer --section local_wood --model DINO \
  --ontology "knot: knot, crack: crack"

# Using Qwen2.5-VL
autolbl-infer --section local_wood --model Qwen \
  --ontology "defect: defect"

# Enable W&B logging
autolbl-infer --section local_wood --model Florence --wandb
```

### 4. Using AutoLbl as a Python Package

```python
from autolbl.models.florence import Florence2
from autolbl.evaluation.metrics import evaluate_detections
from autodistill.detection import CaptionOntology

# Initialize model
ontology = CaptionOntology({
    "knot": "knot",
    "crack": "crack",
    "resin": "resin"
})
model = Florence2(ontology=ontology)

# Run detection
dataset = model.label(
    input_folder="path/to/images",
    output_folder="path/to/output",
    extension=".jpg"
)

# Evaluate results
from supervision import DetectionDataset
gt_dataset = DetectionDataset.from_yolo(
    images_directory_path="path/to/images",
    annotations_directory_path="path/to/annotations",
    data_yaml_path="path/to/data.yaml"
)

confusion_matrix, precision, recall, f1, map50, map50_95 = evaluate_detections(
    dataset, gt_dataset
)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"mAP@50: {map50}")
```

## Supported Models

### Vision-Language Models
- **Florence-2**: Microsoft's vision-language model with object detection capabilities
- **Grounding DINO**: Open-vocabulary object detection with text prompts
- **Qwen2.5-VL**: Alibaba's vision-language model for detection and classification

### Classification Models
- **MetaCLIP**: Image classification using CLIP with metadata

### Composed Models
- **Combined Detection + Classification**: Use one model for detection and another for classification
- Example: Grounding DINO for detection + MetaCLIP for fine-grained classification

## Ontology System

AutoLbl uses a flexible ontology system where you can define classes using natural language:

```python
# Simple ontology
--ontology "defect: defect"

# Multi-class ontology
--ontology "color: color, combined: combined, hole: hole, liquid: liquid, scratch: scratch"

# Descriptive ontology
--ontology "a knot in wood: knot, wood crack: crack, resin pocket: resin"

# Bag-of-words mode (automatic class extraction)
--ontology "BAG_OF_WORDS"
```

## Dataset Preparation

AutoLbl includes comprehensive dataset preparation utilities:

### MVTec AD Wood Dataset
- Converts ground truth masks to YOLO bounding boxes
- Flattens nested directory structure
- Creates empty annotations for defect-free images
- 5 defect classes: color, combined, hole, liquid, scratch

### Zenodo Images1 Dataset
- Converts BMP images to JPG format
- Handles European decimal separators in annotations
- Supports 10+ wood defect classes
- Configurable image limits for testing

See [DATASET_PREPARATION.md](DATASET_PREPARATION.md) for complete documentation.

## Documentation

- **Usage Guide**: See [DATASET_PREPARATION.md](DATASET_PREPARATION.md) for dataset preparation
- **Model Configuration**: See `config.json` for configuration options
- **API Reference**: See code documentation in `utils/` directory

## Project Structure

```
autolbl/
├── run_any3.py                 # Main inference script
├── prepare_datasets.py         # Unified dataset preparation
├── config.json                 # Configuration file
├── DATASET_PREPARATION.md      # Dataset preparation guide
├── utils/                      # Utility modules
│   ├── dataset_preparation.py  # Shared dataset utilities
│   ├── convert_annotations.py  # Annotation conversion functions
│   ├── Florence_fixed.py       # Florence-2 model wrapper
│   ├── grounding_dino_model.py # Grounding DINO wrapper
│   ├── qwen25_model.py         # Qwen2.5-VL wrapper
│   ├── metaclip_model.py       # MetaCLIP wrapper
│   ├── composed_detection_model.py # Model composition
│   ├── check_labels.py         # Evaluation metrics
│   └── wandb_utils.py          # W&B integration
├── data/                       # Dataset storage
└── samples/                    # Output results
```

## Experiment Tracking

AutoLbl integrates with Weights & Biases for experiment tracking:

```bash
# Enable W&B logging
python run_any3.py --section local_wood --model Florence --wandb

# Add custom tags
python run_any3.py --section local_wood --model Florence --wandb --tag wood_defects
```

Logged metrics include:
- Detection visualizations (bounding boxes)
- Confusion matrices
- Precision, recall, F1 scores per class
- mAP scores
- Model hyperparameters

## Examples

### Example 1: Wood Defect Detection
```bash
python run_any3.py --section local_wood --model Florence \
  --ontology "color: color, hole: hole, scratch: scratch" \
  --wandb --tag wood_exp1
```

### Example 2: Single Class Detection
```bash
python run_any3.py --section local_wood --model DINO \
  --ontology "defect: defect" \
  --nms class_agnostic
```

### Example 3: Model Composition
```bash
python run_any3.py --section local_wood --model Combined \
  --ontology "color: color, hole: hole, scratch: scratch"
```

### Example 4: Bag-of-Words Mode
```bash
python run_any3.py --section local_Images1 --model Florence \
  --ontology "BAG_OF_WORDS"
```

## Citations

If you use AutoLbl in your research, please cite the thesis:

```bibtex
@mastersthesis{kolbert2025autolbl,
  title={Automatic Labelling for Image Datasets},
  author={Kolbert, Jędrzej},
  year={2025},
  school={Danmarks Tekniske Universitet},
  type={MSc thesis}
}
```

### Datasets

If you use the MVTec AD dataset:
```bibtex
@inproceedings{bergmann2019mvtec,
  title={MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9592--9600},
  year={2019}
}
```

If you use the Zenodo Images1 dataset:
```bibtex
@dataset{kodytek_pavel_2021_4694695,
  author={Kodytek, Pavel and Bodzas, Alexandra and Bilik, Petr},
  title={{Supporting data for Deep Learning and Machine Vision based approaches 
          for automated wood defect detection and quality control}},
  year={2021},
  publisher={Zenodo},
  doi={10.5281/zenodo.4694695}
}
```

## Todo

- [ ] Add support for segmentation models (SAM integration)
- [ ] Implement active learning pipeline
- [ ] Add model fine-tuning capabilities
- [ ] Support for video/batch processing
- [ ] Web interface for annotation review
- [ ] Docker containerization

## Acknowledgments

This work is built upon the [Roboflow Autodistill](https://github.com/autodistill/autodistill) library, which provides a unified framework for using foundation models to automatically label data for computer vision tasks. We gratefully acknowledge the Roboflow team for making their excellent tools available to the community.

### Autodistill Citation
```bibtex
@misc{autodistill,
  title={Autodistill},
  author={Roboflow},
  year={2023},
  howpublished={\url{https://github.com/autodistill/autodistill}}
}
```

## Contribution

**Note**: This project was developed as part of an MSc thesis and is not currently accepting external contributions. However, issues and feature requests are welcome for discussion and future consideration.

## Contact

If you experience problems or have technical questions, please open an issue on GitHub.

For questions related to the AutoLbl project or thesis work, please contact:
- **Jędrzej Kolbert** - [s184361@dtu.dk](mailto:jedrzej.kolbert@gmail.com)

---

*This project was developed as part of an MSc thesis in Mathematical Modelling and Computation at Danmarks Tekniske Universitet (DTU).*

**Project**: Automatic Labelling for Image Datasets  
**Author**: Jędrzej Kolbert  
**Institution**: DTU - Danmarks Tekniske Universitet  
**Year**: 2024-2025
