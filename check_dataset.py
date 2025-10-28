import os
import json

def check_dataset(config_path="config.json", section="local"):
    """Check if dataset has valid annotations"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    section_config = config.get(section, config.get("local", {}))
    
    images_dir = section_config.get("GT_IMAGES_DIRECTORY_PATH", "")
    annotations_dir = section_config.get("GT_ANNOTATIONS_DIRECTORY_PATH", "")
    yaml_path = section_config.get("GT_DATA_YAML_PATH", "")
    
    print(f"Checking dataset section: {section}")
    print(f"Images directory: {images_dir}")
    print(f"Annotations directory: {annotations_dir}")
    print(f"Data YAML: {yaml_path}")
    print("="*80)
    
    # Check images
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nFound {len(image_files)} image files:")
        for img in image_files[:10]:  # Show first 10
            print(f"  - {img}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more")
    else:
        print(f"\n❌ Images directory does not exist: {images_dir}")
        return
    
    # Check annotations
    if os.path.exists(annotations_dir):
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
        print(f"\nFound {len(annotation_files)} annotation files:")
        
        empty_files = []
        non_empty_files = []
        
        for ann_file in annotation_files[:10]:  # Show first 10
            ann_path = os.path.join(annotations_dir, ann_file)
            with open(ann_path, 'r') as f:
                lines = f.readlines()
                num_lines = len([l for l in lines if l.strip()])
                
            if num_lines == 0:
                empty_files.append(ann_file)
                print(f"  - {ann_file} (EMPTY ❌)")
            else:
                non_empty_files.append(ann_file)
                print(f"  - {ann_file} ({num_lines} annotations ✓)")
        
        if len(annotation_files) > 10:
            print(f"  ... and {len(annotation_files) - 10} more")
        
        print(f"\nSummary:")
        print(f"  Empty annotation files: {len(empty_files)}")
        print(f"  Non-empty annotation files: {len(non_empty_files)}")
        
        if len(non_empty_files) == 0:
            print(f"\n❌ ERROR: All annotation files are empty!")
            print(f"You need to add ground truth annotations to proceed.")
        
    else:
        print(f"\n❌ Annotations directory does not exist: {annotations_dir}")
    
    # Check YAML
    if os.path.exists(yaml_path):
        print(f"\n✓ Data YAML exists")
    else:
        print(f"\n❌ Data YAML does not exist: {yaml_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--section", default="local")
    args = parser.parse_args()
    
    check_dataset(args.config, args.section)