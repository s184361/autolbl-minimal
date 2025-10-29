# Dataset Preparation Guide

This guide explains how to download and prepare datasets for use with AutoLbl.

## Quick Start

The AutoLbl package includes a command-line tool for dataset preparation:

```bash
# Prepare all datasets
autolbl-prepare --dataset all

# Prepare only Wood dataset
autolbl-prepare --dataset wood

# Prepare only Images1 dataset (first 100 images)
autolbl-prepare --dataset images1

# Prepare Images1 with custom image limit
autolbl-prepare --dataset images1 --max-images 500

# Prepare all images from Images1
autolbl-prepare --dataset images1 --max-images 0
```

## Prerequisites

- AutoLbl package installed (`pip install -e .`)
- Internet connection to download the datasets
- Approximately 500 MB of free disk space per dataset

## Supported Datasets

### 1. MVTec AD Wood Dataset
- **Source:** MVTec Anomaly Detection dataset
- **Size:** ~500 MB
- **Images:** 79 test images (1024x1024)
- **Classes:** 5 defect types + good samples

### 2. Images1 Dataset (Zenodo)
- **Source:** Zenodo wood defect dataset
- **Size:** ~5 GB (full dataset)
- **Images:** 20,000+ BMP images
- **Classes:** 10+ wood defect types

## Detailed Setup Instructions

### Wood Dataset (MVTec AD)

#### 1. Download the Wood Dataset

1. Visit the MVTec AD dataset download page:
   ```
   https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
   ```

2. Download the **Wood** dataset (wood.tar.xz)

3. Extract the downloaded archive

#### 2. Place Dataset in Correct Location

Create the data folder structure and place the extracted dataset:
   ```
   autolbl/
   ├── data/
   │   └── wood/
   │       └── wood/          # Extracted MVTec wood dataset
   │           ├── ground_truth/
   │           │   ├── color/
   │           │   ├── combined/
   │           │   ├── hole/
   │           │   ├── liquid/
   │           │   └── scratch/
   │           ├── test/
   │           │   ├── color/
   │           │   ├── combined/
   │           │   ├── good/
   │           │   ├── hole/
   │           │   ├── liquid/
   │           │   └── scratch/
   │           ├── train/
   │           │   └── good/
   │           ├── license.txt
            └── readme.txt
```

#### 3. Run Preparation Script

```bash
autolbl-prepare --dataset wood
```

#### 4. Verify Output

After preparation, you should see:

```
autolbl/
├── data/
│   └── wood/
│       ├── images/              # 79 JPG images
│       │   ├── 000_color.jpg
│       │   ├── 000_combined.jpg
│       │   ├── 000_good.jpg
│       │   ├── 000_hole.jpg
│       │   └── ...
│       ├── yolo_annotations/    # 79 annotation files
│       │   ├── 000_color.txt
│       │   ├── 000_combined.txt
│       │   ├── 000_good.txt
│       │   ├── 000_hole.txt
│       │   └── ...
│       └── data.yaml           # Class definitions
```

Expected output:
- **79 images** in `images/` folder
- **79 annotation files** in `yolo_annotations/` folder
- Empty annotation files for "good" images (no defects)
- Bounding box annotations for defect images

### Images1 Dataset (Zenodo)

#### 1. Download the Dataset

1. Visit Zenodo: https://zenodo.org/records/4694695
2. Download `Images1.zip` and `Bouding_Boxes.zip`

#### 2. Extract and Place Files

Extract the files to the data folder:

Each annotation file contains one line per bounding box:
```
class_id x_center y_center width height
```

Example (`000_color.txt`):
```
0 0.8212890625 0.72607421875 0.068359375 0.1552734375
0 0.4482421875 0.48486328125 0.349609375 0.4365234375
```

All coordinates are normalized (0-1 range).

## Running Experiments

After preparation, use the datasets with `run_any3.py`:

```bash
# Wood dataset
python run_any3.py --section local_wood --model Florence

# Images1 dataset
python run_any3.py --section local_Images1 --model Florence
```

## Troubleshooting

### Issue: "No such file or directory" error

**Solution:** Ensure the datasets are extracted to the correct locations:
- Wood: `data/wood/wood/` (note the nested `wood` folder)
- Images1: `data/Images1/` and `data/Bouding_Boxes/Bouding Boxes/`

### Issue: Wrong number of files generated

**Solution:** 
- Make sure you downloaded the complete dataset
- Extract all files properly
- Check the console output for error messages

### Issue: Script fails with import errors

**Solution:** Install required packages:
```bash
pip install opencv-python numpy supervision
```

### Issue: Images1 processing is too slow

**Solution:** Use the `--max-images` option to process fewer images:
```bash
autolbl-prepare --dataset images1 --max-images 100
```

### Issue: Annotation format errors

**Solution:** The script automatically handles:
- European decimal separators (commas → periods)
- Multiple annotation formats (bounding boxes and masks)
- Missing or empty annotation files

## Dataset Information

### MVTec AD Wood Dataset
- **Category:** Wood texture anomaly detection
- **Total Test Images:** 79
  - 60 good samples (no defects)
  - 19 defect samples across 5 defect types
- **Image Size:** 1024 x 1024 pixels
- **Defect Types:**
  - Color variations
  - Combined defects
  - Holes
  - Liquid residue
  - Scratches
- **License:** See `license.txt` in the dataset

### Images1 Dataset (Zenodo)
- **Category:** Wood defect detection with bounding boxes
- **Total Images:** 20,000+
- **Image Format:** BMP (converted to JPG by script)
- **Annotation Format:** Class name + bounding box coordinates
- **Special Features:**
  - Multiple defect classes
  - European decimal format (comma separators)
  - Varying image sizes

## Citations

### MVTec AD Wood Dataset

If you use the MVTec AD Wood dataset in your research, please cite:

```bibtex
@inproceedings{bergmann2019mvtec,
  title={MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9592--9600},
  year={2019}
}
```

### Images1 Dataset (Zenodo)

If you use the Images1 dataset from Zenodo in your research, please cite:

```bibtex
@dataset{kodytek_pavel_2021_4694695,
  author       = {Kodytek, Pavel and
                  Bodzas, Alexandra and
                  Bilik, Petr},
  title        = {{Supporting data for Deep Learning and Machine 
                   Vision based approaches for automated wood defect
                   detection and quality control}},
  month        = apr,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4694695},
  url          = {https://doi.org/10.5281/zenodo.4694695}
}
```

**Plain text citation:**
> Kodytek Pavel, Bodzas Alexandra, & Bilik Petr. (2021). Supporting data for Deep Learning and Machine Vision based approaches for automated wood defect detection and quality control. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4694695

## Advanced Usage

### Processing Specific Number of Images

Control how many images to process from Images1:

```bash
# First 10 images (quick test)
autolbl-prepare --dataset images1 --max-images 10

# First 1000 images
autolbl-prepare --dataset images1 --max-images 1000

# All images (no limit)
autolbl-prepare --dataset images1 --max-images 0
```

### Reprocessing Datasets

To reprocess a dataset, simply run the script again. It will overwrite existing files.

### Custom Modifications

All shared utility functions are in `utils/dataset_preparation.py`:
- `get_base_dir()` - Auto-detect project root
- `update_config_section()` - Update configuration
- `convert_bbox_annotation()` - Handle annotation formats
- `convert_masks_to_yolo_bbox()` - Extract bounding boxes from masks
- `copy_and_rename_images()` - Flatten directory structures
- `create_empty_annotations()` - Create empty files for defect-free images

## Next Steps

After preparing the dataset(s), you can:

1. Use the `local_wood` or `local_Images1` configuration in your experiments
2. Train object detection models on the defect datasets
3. Evaluate model performance using the ground truth annotations
4. Compare results across different datasets and models

For more information on running experiments, refer to the main project README.
