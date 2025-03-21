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
    def __init__(self, img_folder, initial_prompt="wood defects irregularities", indices=None):
        self.indices = indices  # Also fix the instance variable name
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.ontology = CaptionOntology({"defect": "defect"})
        self.base_model = MetaCLIP(self.ontology)
        self.optimizer = "COBYLA"
        
        # Load and embed all images from folders
        self.embs = []
        
        # Process good images
        print(f"Images from {img_folder}...")
        good_img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if self.indices is not None:
            good_img_files = [good_img_files[i] for i in self.indices]
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
    
    def objective(self, x, img=None):
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
        if img is not None: #for new images
            good_similarities = self.base_model.compare(emb, prompt_embedding)
        else:
            good_similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.embs]
        #bad_similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.bad_embs]
        
        avg_good_similarity = np.mean(good_similarities)
        #avg_bad_similarity = np.mean(bad_similarities)
        
        # Our goal: maximize ratio of bad/good similarity scores
        # We want a prompt that distinguishes defects well
        ratio = avg_good_similarity#-avg_bad_similarity + 1e-6)  # Add epsilon to avoid division by zero
        
        #print(f"Prompt: '{prompt}', Good sim: {avg_good_similarity:.4f}, Bad sim: {avg_bad_similarity:.4f}, Ratio: {ratio:.4f}")
        return -ratio  # Lower is better
    
    def optimize(self, maxiter=50, optimizer="differential_evolution", img=None):
        """Optimize the prompt using differential evolution or COBYLA"""
        self.optimizer = optimizer
        x0 = self.input_ids.detach().numpy()
        bounds = [(0, len(self.tokenizer))] * len(x0)
        
        print(f"Starting optimization with {len(x0)} tokens using {optimizer}...")
        
        if self.optimizer == "COBYLA":
            result = minimize(
            lambda x: self.objective(x, img=img), 
            x0, 
            method="COBYLA", 
            options={'maxiter': maxiter, 'rhobeg': 100.0, 'tol': 1e-6},
            bounds=bounds
            )
        elif self.optimizer == "differential_evolution":
            result = differential_evolution(
            lambda x: self.objective(x, img=img), 
            bounds, 
            popsize=15, 
            mutation=(0.5, 1.5),
            recombination=0.7, 
            maxiter=maxiter
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
            
        # Get the optimized prompt
        optimized_tensor = torch.tensor(result.x, dtype=torch.float32)
        optimized_prompt = self.decode_prompt(optimized_tensor)
        
        print("\nOptimization Results:")
        print(f"Final ratio: {result.fun}")
        print(f"Optimized prompt: '{optimized_prompt}'")
        if img is not None:
            return optimized_prompt
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
    path = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/autolbl"
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    vectorized = True
    for bad_fldr in ["color","combined", "good", "hole", "liquid","scratch"]:
        torch.cuda.empty_cache()
        # Paths to your image folders

        good_img_folder = os.path.join(path, "data/wood/test/good")
        bad_img_folder = os.path.join(path, f"data/wood/test/{bad_fldr}")  # assuming "color" contains defect images

        # Create optimizer and run optimization
        optimizer_good = PromptOptimizer(
            img_folder=good_img_folder,
            initial_prompt="wood",
            indices=indices,
        )
        good_prompt = optimizer_good.optimize(
            maxiter=1000, optimizer="differential_evolution"
        )

        # prepare test data
        test_files = [good_img_folder+"/"+ f for f in os.listdir(good_img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # filter out the images that were used for training
        test_files = [test_files[i] for i in range(len(test_files)) if i not in indices]
        labels = np.zeros(len(test_files))
        # append bad images
        test_files.extend([bad_img_folder+"/"+f for f in os.listdir(bad_img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        labels = np.append(labels, np.ones(len(test_files)-len(labels)))

        # prepare good differences
        #good_prompt_emb = optimizer_good.base_model.embed_text(good_prompt)
        if vectorized:
            good_differences=[]
            good_files = [f for f in os.listdir(good_img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            good_files = [good_files[i] for i in indices]
            good_prompt_emb = torch.zeros((len(good_files), 512))
            for i in range(len(good_files)):
                img_path = os.path.join(good_img_folder, good_files[i])
                img = PIL.Image.open(img_path)
                good_prompt=optimizer_good.optimize(
                    maxiter=1000, optimizer="differential_evolution", img=img
                    )
                good_prompt_emb[i] = optimizer_good.base_model.embed_text(good_prompt)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                good_differences.append(optimizer_good.base_model.vec_difference(optimizer_good.embs[i].to(device), good_prompt_emb[i].to(device)).cpu())
        else:
            good_differences = [
                optimizer_good.base_model.difference(emb, good_prompt_emb)
                for emb in optimizer_good.embs
            ]

        diff_test = np.zeros((len(test_files)))
        if vectorized:
            bad_difference = torch.zeros((len(test_files), 512))
        else:
            bad_difference = np.zeros((len(test_files)))
        for i in range(len(test_files)):
            img_path = os.path.join(good_img_folder, test_files[i])
            img = PIL.Image.open(img_path)
            emb = optimizer_good.base_model.embed_image(img)
            prompt_bad = optimizer_good.optimize(
                maxiter=1000, optimizer="differential_evolution", img=img
            )
            if vectorized:
                bad_difference[i] = optimizer_good.base_model.vec_difference(emb, optimizer_good.base_model.embed_text(prompt_bad)).cpu()

                # Convert list of tensors to a stacked tensor
                good_diffs_stacked = torch.stack([gd.cpu() for gd in good_differences])

                # Calculate differences with broadcasting
                diffs = torch.abs(bad_difference[i].unsqueeze(0) - good_diffs_stacked)
                diff_test[i] = torch.median(diffs).item()#torch.max(diffs).item()
            else:
                bad_difference[i] = optimizer_good.base_model.difference(emb, optimizer_good.base_model.embed_text(prompt_bad)).cpu()
                diff_test[i] = np.max(np.abs(bad_difference[i] - np.array(good_differences)), axis=0)
            torch.cuda.empty_cache()

        # plot the results
        if vectorized:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(labels, diff_test)
        ax1.boxplot([diff_test[labels == 0], diff_test[labels == 1]], positions=[0, 1])
        ax1.set_ylabel("Difference between diffence for test and diffrece for train")
        ax1.set_xlabel("True Label")
        ax1.set_xticklabels(["Good Test", "Bad Test"])
        ax1.set_title(bad_img_folder.split("/")[-1])

        if vectorized:
            bad_difference = bad_difference.cpu().numpy()
            im = ax2.imshow(bad_difference, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax2, label="Difference value")
            ax2.set_ylabel("Test image index")
            ax2.set_xlabel("Embedding dimension")
            ax2.set_title(f"Embedding differences - {bad_img_folder.split('/')[-1]}")

            # Explicitly squeeze the tensor to remove the extra dimension
            good_diffs_np = good_diffs_stacked.cpu().numpy()[:, 0, :]
            im3 = ax3.imshow(good_diffs_np, aspect='auto', cmap='viridis')
            plt.colorbar(im3, ax=ax3, label="Difference value")
            ax3.set_ylabel("Train image index")
            ax3.set_xlabel("Embedding dimension")
            ax3.set_title(f"Embedding differences - Good Train Images")
            
            # Ensure both plots have the same x-axis limits
            ax2.set_xlim(0, 512)  # Assuming embedding dimension is 512
            ax3.set_xlim(0, 512)
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
        else:
            ax2.scatter(labels, bad_difference)
            ax2.scatter(np.ones(len(good_differences))*2, good_differences, label="good")
            ax2.boxplot([bad_difference[labels == 0], bad_difference[labels == 1], good_differences], positions=[0, 1, 2])
            ax2.set_ylabel("Difference between prompt and image")
            ax2.set_xlabel("True Label")
            ax2.set_xticklabels(["Good Test", "Bad Train", "Good Train"])
        ax2.set_title(bad_img_folder.split("/")[-1])
        plt.savefig(f"results/{bad_img_folder.split('/')[-1]}_results_vectorized_ind_good_prpt_median.png")
