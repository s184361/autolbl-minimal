import torch
from transformers import BertTokenizerFast
from scipy.optimize import differential_evolution
import numpy as np
from utils.metaclip_model_classifier import MetaCLIP
from autodistill.detection import CaptionOntology
import PIL
import os
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PromptOptimizer:
    def __init__(self, img_folder, initial_prompt="wood defects irregularities"):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.ontology = CaptionOntology({"defect": "defect"})
        self.base_model = MetaCLIP(self.ontology)
        self.optimizer = "COBYLA"
        
        # Load and embed all images from folders
        self.embs = []
        
        # Process good images
        print(f"Images from {img_folder}...")
        good_img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in tqdm(good_img_files):
            img_path = os.path.join(img_folder, img_file)
            try:
                img = PIL.Image.open(img_path)
                emb = self.base_model.embed_image(img)
                self.embs.append(emb)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
        
        print(f"Loaded {len(self.embs)} images")
        
        # Initial prompt
        self.initial_prompt = initial_prompt
        input_ids = self.tokenizer(initial_prompt)['input_ids']
        self.input_ids = torch.tensor(input_ids[1:-1], dtype=torch.float32)  # Remove CLS and SEP tokens
        print(f"Initial prompt: '{initial_prompt}'")
        print(f"Token IDs: {self.input_ids}")
        
    def decode_prompt(self, input_ids):
        """Convert token IDs back to text prompt"""
        rounded_ids = torch.round(input_ids).clamp(0, len(self.tokenizer)).to(torch.int64)
        prompt = self.tokenizer.decode(rounded_ids, skip_special_tokens=True)
        return prompt
    
    def objective(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        prompt = self.decode_prompt(x_tensor)
        
        # Stricter validation of prompts
        if not prompt.strip() or len(prompt.split()) < 2:
            return 0.0
        
        try:
            # Force CPU execution to avoid CUDA errors
            prompt_embedding = self.base_model.embed_text(prompt)
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")
            return 0.0
        
        # Get text embedding for the current prompt
        prompt_embedding = self.base_model.embed_text(prompt)
        
        # Calculate average similarity scores for all good and bad images
        good_similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.embs]
        #bad_similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.bad_embs]
        
        avg_good_similarity = np.mean(good_similarities)
        #avg_bad_similarity = np.mean(bad_similarities)
        
        # Our goal: maximize ratio of bad/good similarity scores
        # We want a prompt that distinguishes defects well
        ratio = avg_good_similarity#-avg_bad_similarity + 1e-6)  # Add epsilon to avoid division by zero
        
        #print(f"Prompt: '{prompt}', Good sim: {avg_good_similarity:.4f}, Bad sim: {avg_bad_similarity:.4f}, Ratio: {ratio:.4f}")
        return -ratio  # Lower is better
    
    def optimize(self, maxiter=50, optimizer="differential_evolution"):
        """Optimize the prompt using differential evolution or COBYLA"""
        self.optimizer = optimizer
        x0 = self.input_ids.detach().numpy()
        bounds = [(0, len(self.tokenizer))] * len(x0)
        
        print(f"Starting optimization with {len(x0)} tokens using {optimizer}...")
        
        if self.optimizer == "COBYLA":
            result = minimize(self.objective, x0, method="COBYLA", 
                              options={'maxiter': maxiter, 'rhobeg': 100.0, 'tol': 1e-6},
                              bounds=bounds)
        elif self.optimizer == "differential_evolution":
            result = differential_evolution(self.objective, bounds, 
                               popsize=15, mutation=(0.5, 1.5),
                               recombination=0.7, maxiter=maxiter)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
            
        # Get the optimized prompt
        optimized_tensor = torch.tensor(result.x, dtype=torch.float32)
        optimized_prompt = self.decode_prompt(optimized_tensor)
        
        print("\nOptimization Results:")
        print(f"Final ratio: {result.fun}")
        print(f"Optimized prompt: '{optimized_prompt}'")
        
        # Test the optimized prompt
        prompt_embedding = self.base_model.embed_text(optimized_prompt)
        
        similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.embs]
        #bad_similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.bad_embs]
        
        avg_similarity = np.mean(similarities)
        #avg_bad_similarity = np.mean(bad_similarities)
        
        print(f"Average image similarity: {avg_similarity:.4f}")
        #print(f"Average bad image similarity: {avg_bad_similarity:.4f}")
        #print(f"Ratio (good/bad): {avg_good_similarity/avg_bad_similarity:.4f}")
        #print(f"Inverse ratio (bad/good): {avg_bad_similarity/avg_good_similarity:.4f}")
        
        return optimized_prompt

if __name__ == "__main__":
    # Paths to your image folders
    path = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/autolbl"
    good_img_folder = os.path.join(path, "data/wood/test/good")
    bad_img_folder = os.path.join(path, "data/wood/test/color")  # assuming "color" contains defect images

    # Create optimizer and run optimization
    optimizer_good = PromptOptimizer(
        img_folder=good_img_folder,
        initial_prompt="wood",
    )
    good_prompt = optimizer_good.optimize(
        maxiter=1000, optimizer="differential_evolution"
    )
    optimizer_bad = PromptOptimizer(
        img_folder=bad_img_folder,
        initial_prompt="wood",
    )
    bad_prompt = optimizer_bad.optimize(
        maxiter=1000, optimizer="differential_evolution"
    )

    print(f"Good prompt: '{good_prompt}'")
    print(f"Bad prompt: '{bad_prompt}'")

    # calculate good simlarities for good prompt and good images
    good_prompt_emb =optimizer_good.base_model.embed_text(good_prompt)
    bad_prompt_emb = optimizer_good.base_model.embed_text(bad_prompt)
    good_similarities = [ optimizer_good.base_model.compare(emb, good_prompt_emb) for emb in optimizer_good.embs]
    good_differences = [ optimizer_good.base_model.difference(emb, good_prompt_emb) for emb in optimizer_good.embs]
    print(f"Good differences: {good_differences}")
    print(f"Good similarities: {good_similarities}")
    
    # calculate good simlarities for bad prompt and good images
    bad_similarities = [ optimizer_good.base_model.compare(emb, optimizer_good.base_model.embed_text(bad_prompt)) for emb in optimizer_good.embs]
    bad_differences = [ optimizer_good.base_model.difference(emb, bad_prompt_emb) for emb in optimizer_good.embs]
    print(f"Bad differences: {bad_differences}")
    print(f"Bad similarities: {bad_similarities}")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # First subplot for similarities
    ax1.hist(good_similarities, bins=20, alpha=0.5, label="good")
    ax1.hist(bad_similarities, bins=20, alpha=0.5, label="bad")
    ax1.set_title('Similarities Distribution')
    ax1.legend(loc="upper right")
    
    # Second subplot for differences
    ax2.hist(good_differences, bins=20, alpha=0.5, label="good")
    ax2.hist(bad_differences, bins=20, alpha=0.5, label="bad")
    ax2.set_title('Differences Distribution')
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()

    diff_good_bad = np.empty(len(bad_differences))
    for i in range(len(bad_differences)):
        dd = np.abs(bad_differences[i] - np.array(good_differences))
        diff_good_bad[i] = np.max(dd)

    print(f"Difference good-bad: {diff_good_bad}")
    diff_good_good = np.empty(len(good_differences))
    for i in range(len(good_differences)):
        dd = np.abs(good_differences[i] - np.array(good_differences))
        diff_good_good[i] = np.max(dd)

    print(f"Difference good-good: {diff_good_good}")

    #both in the same plot
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.hist(diff_good_bad, bins=20, alpha=0.5, label="good-bad")
    ax.hist(diff_good_good, bins=20, alpha=0.5, label="good-good")
    ax.set_title('Differences Distribution')
    ax.legend(loc="upper right")
    plt.show()
