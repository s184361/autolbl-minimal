# %%
import os
import clip
import torch
from torchvision.datasets import CIFAR100
import re
from PIL import Image
import matplotlib.pyplot as plt
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
# %%
# Step 1: Open the file and read its contents
with open("data/Semantic Map Specification.txt", "r") as file:
    content = file.read()
names = re.findall(r"name=([^\n]+)", content)
# lowercase names and remove _
names = [name.lower().replace("_", " ") for name in names]
# %%
# Bounding Boxes/100000001_anno.txt
# Images1/100000001.bmp
image = Image.open("data/Images1/100000001.bmp")
# Prepare the inputs

image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat(
    # [clip.tokenize(f"a photo of a {c}") for c in names]
    [clip.tokenize(f"a photo of a {c} wood defect") for c in names]
).to(device)
# %%
# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{names[index]:>16s}: {100 * value.item():.2f}%")
# show the image

# %%
