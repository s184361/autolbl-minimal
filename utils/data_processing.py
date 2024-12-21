import yaml
import os
from config import HOME, GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH  # Explicit imports

def defect_annotations(gt_annotations_directory_path, gt_data_yaml_path, defect_annotation_path):
    """
    Create annotation files containing only the defect class.

    Args:
        gt_annotations_directory_path: Path to the ground truth annotations directory.
        gt_data_yaml_path: Path to the YAML file containing class names.
        defect_annotation_path: Path to save the new defect-only annotation files.
    """
    # Load class names from YAML
    with open(gt_data_yaml_path, "r") as file:
        data_yaml = yaml.safe_load(file)
    class_names = data_yaml["names"]

    # Ensure the defect_annotation_path exists
    os.makedirs(defect_annotation_path, exist_ok=True)

    # Iterate through annotation files in the ground truth annotations directory
    for filename in os.listdir(gt_annotations_directory_path):
        if filename.endswith(".txt") and not filename.startswith("confidence-"):
            file_path = os.path.join(gt_annotations_directory_path, filename)

            # Read the annotation file
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Filter annotations to keep only defect classes
            defect_lines = []
            for line in lines:
                #replace the first part with 1
                parts = line.strip().split()
                if len(parts) > 0:
                    # Assuming the defect class is represented by the first class in the class_names list
                    parts[0] = "1"  # Replace the class ID with 1 for defect class
                    defect_lines.append(" ".join(parts) + "\n")

            # Save the new defect-only annotation file
            if defect_lines:
                new_file_path = os.path.join(defect_annotation_path, filename)
                with open(new_file_path, "w") as new_file:
                    new_file.writelines(defect_lines)

def main():
    def_anno_path = f"{HOME}/data/defect_anno"
    defect_annotations(GT_ANNOTATIONS_DIRECTORY_PATH, GT_DATA_YAML_PATH, def_anno_path)
if __name__ == "__main__":
    main()
