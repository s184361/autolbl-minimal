#%%
import os

import matplotlib.pyplot as plt
import os
import random
from PIL import Image

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
!python ./OpenGroundingDino/tools/coco2odvg.py --input "/aquarium_data/train/_annotations.coco.json"  --output "/input_params/train.jsonl"
# %%
