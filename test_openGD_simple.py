#%%
import os

import matplotlib.pyplot as plt
import os
import random
import json
from PIL import Image
import re

from transformers import AutoTokenizer, AutoModel
#%%
# Path to the folder containing the images
folder_path = os.getcwd() + '/aquarium_data/train'

# Get a list of all files in the folder
all_files = os.listdir(folder_path)

# Filter the list to include only files with image extensions
image_files = [file for file in all_files if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]

# Randomly select 16 images from the list
selected_images = random.sample(image_files, 16)

# Set up the plot grid
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

# Plot each selected image in the grid
for ax, image_file in zip(axes.flatten(), selected_images):
    # Open the image file
    img = Image.open(os.path.join(folder_path, image_file))

    # Display the image on the grid
    ax.imshow(img)
    ax.axis('off')  # Hide the axes

# Adjust the layout to be tight
plt.tight_layout()

# Show the plot
plt.show()
#%%
import re

# Define the file path
file_path = os.getcwd()+ '/OpenGroundingDino/tools/coco2odvg.py'

# Define the new values according to the dataset
new_id_map = '{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}'# 7 classes of AquariumDataset
new_ori_map = '{"1": "fish", "2": "jellyfish", "3": "penguins", "4": "sharks", "5": "puffins", "6":"stingrays", "7": "starfish"}'

# Read the content of the file
with open(file_path, 'r') as file:
    content = file.read()

# Replace the id_map value using regex
content = re.sub(r'id_map\s*=\s*\{[^\}]*\}', f'id_map = {new_id_map}', content)

# Replace the ori_map value using regex
content = re.sub(r'ori_map\s*=\s*\{[^\}]*\}', f'ori_map = {new_ori_map}', content)

# Write the updated content back to the file
with open(file_path, 'w') as file:
    file.write(content)

print(f"Updated {file_path} successfully.")

#%%
os.makedirs("input_params", exist_ok=True)
!python ./OpenGroundingDino/tools/coco2odvg.py --input "/zhome/4a/b/137804/Desktop/autolbl/aquarium_data/train/_annotations.coco.json"  --output "/zhome/4a/b/137804/Desktop/autolbl/input_params/train.jsonl"
# %%
# Define the content for the JSON file
content = {
    "0": "fish",
    "1": "jellyfish",
    "2": "penguins",
    "3": "sharks",
    "4": "puffins",
    "5": "stingrays",
    "6": "starfish",
}

# Define the file path
file_path = os.getcwd()+'/input_params/label.json'

# Write the content to the JSON file
with open(file_path, 'w') as file:
    json.dump(content, file)

print(f"File '{file_path}' created successfully.")
# %%
# Define the data
data = {
    "train": [
        {
            "root": os.path.join(os.getcwd(), "aquarium_data/train"),  # Train images
            "anno": os.path.join(os.getcwd(), "input_params/train.jsonl"),  # Odvg jsonl file
            "label_map": os.path.join(os.getcwd(), "input_params/label.json"),  # label.json file
            "dataset_mode": "odvg"
        }
    ],
    "val": [
        {
            "root": os.path.join(os.getcwd(), "aquarium_data/test"),  # Test Images
            "anno": os.path.join(os.getcwd(), "aquarium_data/test/_annotations.coco.json"),  # Test data Annotation file
            "label_map": None,
            "dataset_mode": "coco"
        }
    ]
}

# Create the config directory if it doesn't exist
config_dir = os.path.join(os.getcwd(), "OpenGroundingDino", "config")
os.makedirs(config_dir, exist_ok=True)

# Define the file path relative to current working directory
file_path = os.path.join(config_dir, "datasets_mixed_odvg.json")

# Write the content to the JSON file
with open(file_path, 'w') as file:
    json.dump(data, file, indent=2)

print(f"Data has been written to {file_path}")

# %%

def modify_file(file_path):
    label_list_content = 'label_list = ["fish","jellyfish","penguins","sharks","puffins","stingrays","starfish"]\n'

    # Read the entire content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace use_coco_eval =TRUE with use_coco_eval =FALSE using regex
    content = re.sub(r'use_coco_eval\s*=\s*True', 'use_coco_eval = False', content)

    # Insert label_list after use_coco_eval = FALSE using regex
    content = re.sub(r'use_coco_eval\s*=\s*False', r'use_coco_eval = False\n\n' + label_list_content, content, count=1, flags=re.MULTILINE)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)

# Paths to the files
cfg_coco_path = os.getcwd()+'/OpenGroundingDino/config/cfg_coco.py'
cfg_odvg_path = os.getcwd()+'/OpenGroundingDino/config/cfg_odvg.py'

# Modify both files
modify_file(cfg_coco_path)
modify_file(cfg_odvg_path)

print("Updated use_coco_eval to FALSE and added label_list using regex in both files.")

# %%
#make a output directory to store the checkpoints of trained model
os.makedirs(os.getcwd()+"/output", exist_ok=True)

#%%


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
# %%
