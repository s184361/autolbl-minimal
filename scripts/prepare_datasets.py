"""
Unified dataset preparation script.
Handles multiple datasets: Wood (MVTec AD) and Images1 (Zenodo).
"""

import argparse
from pathlib import Path
from autolbl.datasets.dataset_prep import (
    get_base_dir, update_config_section, create_config_section,
    convert_bbox_annotation, convert_masks_to_yolo_bbox,
    copy_and_rename_images, create_empty_annotations,
    create_data_yaml
)
import cv2
import os


def prepare_wood_dataset(base_dir, max_images=None):
    """
    Prepare MVTec AD Wood dataset.
    
    Args:
        base_dir: Base directory of the project
        max_images: Maximum number of images to process (None for all)
    """
    print("="*60)
    print("Preparing MVTec AD Wood Dataset")
    print("="*60)
    
    # Define paths
    source_base = base_dir / "data" / "wood" / "wood"
    output_base = base_dir / "data" / "wood"
    
    images_output_dir = output_base / "images"
    annotations_output_dir = output_base / "yolo_annotations"
    
    images_output_dir.mkdir(parents=True, exist_ok=True)
    annotations_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSource: {source_base}")
    print(f"Output: {output_base}\n")
    
    # Check if source exists
    if not source_base.exists():
        print(f"❌ ERROR: Source directory not found: {source_base}")
        print("Please extract wood.tar.xz to data/wood/wood/")
        return False
    
    # Step 1: Convert masks to YOLO annotations
    ground_truth_dir = source_base / "ground_truth"
    if ground_truth_dir.exists():
        print("Converting ground truth masks to YOLO annotations...")
        classes = {
            'color': 0,
            'combined': 1,
            'hole': 2,
            'liquid': 3,
            'scratch': 4
        }
        convert_masks_to_yolo_bbox(str(ground_truth_dir), str(annotations_output_dir), classes)
        print("✓ Mask conversion complete\n")
    
    # Step 2: Process TEST folder images
    test_dir = source_base / "test"
    if test_dir.exists():
        print("Processing test folder images...")
        count = copy_and_rename_images(str(test_dir), str(images_output_dir), append_folder_name=True)
        print(f"✓ Processed {count} images\n")
        
        print("Creating annotations for good images...")
        good_folder = test_dir / "good"
        if good_folder.exists():
            good_count = create_empty_annotations(str(good_folder), str(annotations_output_dir), suffix='good')
            print(f"✓ Created {good_count} empty annotations for good images\n")
    
    # Step 3: Create data.yaml
    data_yaml_path = output_base / "data.yaml"
    class_names = ['color', 'combined', 'hole', 'liquid', 'scratch']
    create_data_yaml(data_yaml_path, class_names)
    
    # Step 4: Update config.json
    config_section = create_config_section(
        base_dir=base_dir,
        dataset_name='local_wood',
        images_dir=images_output_dir,
        annotations_dir=annotations_output_dir,
        data_yaml_path=data_yaml_path,
        prompt="defects are irregularities that can affect the appearance, strength, or utility"
    )
    
    update_config_section('local_wood', config_section)
    
    print("\n" + "="*60)
    print("✓ Wood dataset preparation complete!")
    print("="*60)
    print(f"\nRun experiments with:")
    print(f"  python run_any3.py --section local_wood --model Florence")
    
    return True


def prepare_images1_dataset(base_dir, max_images=100):
    """
    Prepare Images1 dataset from Zenodo.
    
    Args:
        base_dir: Base directory of the project
        max_images: Maximum number of images to process (None for all)
    """
    print("="*60)
    print("Preparing Images1 Dataset (Zenodo)")
    print("="*60)
    
    # Define paths
    source_images_dir = base_dir / "data" / "Images1"
    source_annotations_dir = base_dir / "data" / "Bouding_Boxes" / "Bouding Boxes"
    
    output_base = base_dir / "data" / "Images1_processed"
    output_images_dir = output_base / "images"
    output_annotations_dir = output_base / "yolo_annotations"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSource images: {source_images_dir}")
    print(f"Source annotations: {source_annotations_dir}")
    print(f"Output: {output_base}\n")
    
    # Check if sources exist
    if not source_images_dir.exists():
        print(f"❌ ERROR: Images directory not found: {source_images_dir}")
        print("Please extract Images1.zip to data/Images1/")
        return False
    
    if not source_annotations_dir.exists():
        print(f"❌ ERROR: Annotations directory not found: {source_annotations_dir}")
        print("Please extract Bouding_Boxes.zip to data/Bouding_Boxes/")
        return False
    
    # Get list of images
    image_files = list(source_images_dir.glob("*.bmp"))
    print(f"Found {len(image_files)} BMP images")
    
    if max_images is not None and len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"Processing first {max_images} images only\n")
    
    # Collect class names
    class_names_set = set()
    processed_count = 0
    skipped_count = 0
    
    print("Converting images and annotations...")
    for image_file in image_files:
        base_name = image_file.stem
        annotation_file = source_annotations_dir / f"{base_name}_anno.txt"
        
        if not annotation_file.exists():
            skipped_count += 1
            continue
        
        # Convert image from BMP to JPG
        image = cv2.imread(str(image_file))
        if image is None:
            skipped_count += 1
            continue
        
        image_height, image_width = image.shape[:2]
        output_image_path = output_images_dir / f"{base_name}.jpg"
        cv2.imwrite(str(output_image_path), image)
        
        # Read and convert annotations
        yolo_annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                result = convert_bbox_annotation(line, image_width, image_height)
                if result is None:
                    continue
                
                class_name, x_center, y_center, width, height = result
                class_names_set.add(class_name)
                yolo_annotations.append((class_name, x_center, y_center, width, height))
        
        # Save annotations temporarily with class names
        output_annotation_path = output_annotations_dir / f"{base_name}.txt"
        with open(output_annotation_path, 'w') as f:
            for ann in yolo_annotations:
                class_name, x_center, y_center, width, height = ann
                f.write(f"{class_name} {x_center} {y_center} {width} {height}\n")
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"  Processed {processed_count} images...")
    
    print(f"\n✓ Image conversion complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images\n")
    
    # Create class mapping
    class_names = sorted(list(class_names_set))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Found {len(class_names)} classes:")
    for idx, name in enumerate(class_names[:10]):  # Show first 10
        print(f"  {idx}: {name}")
    if len(class_names) > 10:
        print(f"  ... and {len(class_names) - 10} more")
    
    # Update annotations with class IDs
    print("\nUpdating annotations with class IDs...")
    annotation_files = list(output_annotations_dir.glob("*.txt"))
    
    for annotation_file in annotation_files:
        updated_lines = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_name = parts[0]
                class_id = class_to_id[class_name]
                x_center, y_center, width, height = parts[1:5]
                
                updated_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        with open(annotation_file, 'w') as f:
            f.writelines(updated_lines)
    
    print("✓ Annotations updated\n")
    
    # Create data.yaml
    data_yaml_path = output_base / "data.yaml"
    create_data_yaml(data_yaml_path, class_names)
    
    # Update config.json
    config_section = create_config_section(
        base_dir=base_dir,
        dataset_name='local_Images1',
        images_dir=output_images_dir,
        annotations_dir=output_annotations_dir,
        data_yaml_path=data_yaml_path,
        prompt="defects spots anomalies misscoloration irregularities"
    )
    
    update_config_section('local_Images1', config_section)
    
    print("\n" + "="*60)
    print("✓ Images1 dataset preparation complete!")
    print("="*60)
    print(f"\nRun experiments with:")
    print(f"  python run_any3.py --section local_Images1 --model Florence")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for object detection experiments"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['wood', 'images1', 'all'],
        default='all',
        help='Dataset to prepare: wood (MVTec AD), images1 (Zenodo), or all'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (default: all for wood, 100 for images1)'
    )
    
    args = parser.parse_args()
    
    # Get base directory
    base_dir = get_base_dir(__file__)
    
    print(f"\nBase directory: {base_dir}\n")
    
    success = True
    
    if args.dataset in ['wood', 'all']:
        if not prepare_wood_dataset(base_dir, args.max_images):
            success = False
        print()
    
    if args.dataset in ['images1', 'all']:
        max_imgs = args.max_images if args.max_images is not None else 100
        if not prepare_images1_dataset(base_dir, max_imgs):
            success = False
        print()
    
    if success:
        print("\n🎉 All datasets prepared successfully!")
    else:
        print("\n⚠️  Some datasets could not be prepared. Check messages above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
