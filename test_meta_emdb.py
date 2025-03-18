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

class PromptOptimizer:
    def __init__(self, good_img_path, bad_img_path, initial_prompt="wood defects irregularities"):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.ontology = CaptionOntology({"defect": "defect"})
        self.base_model = MetaCLIP(self.ontology)
        self.optimizer = "COBYLA"
        # Load and embed images
        self.good_img = PIL.Image.open(good_img_path)
        self.bad_img = PIL.Image.open(bad_img_path)
        self.good_emb = self.base_model.embed_image(self.good_img)
        self.bad_emb = self.base_model.embed_image(self.bad_img)
        
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
        """Objective function for optimization"""
        x_tensor = torch.tensor(x, dtype=torch.float32)
        prompt = self.decode_prompt(x_tensor)
        
        # Skip empty prompts
        if not prompt.strip():
            return 0
            
        # Get text embedding for the current prompt
        prompt_embedding = self.base_model.embed_text(prompt)
        
        # Calculate similarity scores
        good_similarity = self.base_model.compare(self.good_emb, prompt_embedding)
        bad_similarity = self.base_model.compare(self.bad_emb, prompt_embedding)
        
        # Our goal: maximize similarity with bad (defect) images and minimize with good images
        # We want a prompt that can distinguish defects
        loss = -good_similarity/(bad_similarity+1e-6)
        
        #print(f"Prompt: '{prompt}', Good sim: {good_similarity:.4f}, Bad sim: {bad_similarity:.4f}, Loss: {loss:.4f}")
        return loss  # Lower is better
    
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
        print(f"Final loss: {result.fun}")
        print(f"Optimized prompt: '{optimized_prompt}'")
        
        # Test the optimized prompt
        prompt_embedding = self.base_model.embed_text(optimized_prompt)
        good_similarity = self.base_model.compare(self.good_emb, prompt_embedding)
        bad_similarity = self.base_model.compare(self.bad_emb, prompt_embedding)
        
        print(f"Good image similarity: {good_similarity:.4f}")
        print(f"Bad image similarity: {bad_similarity:.4f}")
        print(f"Difference: {bad_similarity - good_similarity:.4f}")
        
        return optimized_prompt

if __name__ == "__main__":
    # Paths to your images
    path = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/autolbl"
    good_img_path = os.path.join(path, "data/wood/test/good/000.png")
    bad_img_path = os.path.join(path, "data/wood/test/color/000.png")
    
    # Create optimizer and run optimization
    optimizer = PromptOptimizer(
        good_img_path=good_img_path,
        bad_img_path=bad_img_path,
        initial_prompt="[PAD] clean [PAD] wood [PAD] [PAD]"
    )
    optimized_prompt = optimizer.optimize(maxiter=10000, optimizer="COBYLA")
