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
import cma

# Add at the beginning of your script
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        #clear the cache
        torch.cuda.empty_cache()
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
        if not prompt.strip() or len(prompt.split()) < 1:
            return 0.0

        # Get text embedding for the current prompt
        try:
            prompt_embedding = self.base_model.embed_text(prompt)
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")
            return 0.0

        # Calculate average similarity scores for all good and bad images
        if img is not None: #for new images
            emb = self.base_model.embed_image(img)
            good_similarities = self.base_model.compare(emb, prompt_embedding)
        else:
            good_similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.embs]
        # bad_similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.bad_embs]

        avg_good_similarity = np.mean(good_similarities)
        # avg_bad_similarity = np.mean(bad_similarities)

        # Our goal: maximize ratio of bad/good similarity scores
        # We want a prompt that distinguishes defects well
        ratio = avg_good_similarity#-avg_bad_similarity + 1e-6)  # Add epsilon to avoid division by zero

        # print(f"Prompt: '{prompt}', Good sim: {avg_good_similarity:.4f}, Bad sim: {avg_bad_similarity:.4f}, Ratio: {ratio:.4f}")
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
            popsize=10,  # Reduced from 15
            mutation=(0.3, 1.0),  # More conservative
            recombination=0.7, 
            maxiter=maxiter
            )
        elif self.optimizer == "CMA":
            # CMA-ES optimization
            sigma0 = 10000.0  # Initial step size
            
            # Run optimization with CMA-ES
            cma_result = cma.fmin(
                lambda x: self.objective(x, img=img),
                x0,
                sigma0,
                options={
                    'bounds': [0, len(self.tokenizer)],
                    'maxfevals': maxiter * len(x0) * 10,  # Adjust based on dimensionality
                    'verbose': 1  # Show progress
                }
            )
            
            # Create result object with similar structure to other optimizers
            class Result:
                def __init__(self, x, fun):
                    self.x = x
                    self.fun = fun
            
            result = Result(cma_result[0], cma_result[1])  # xbest, fbest
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
        # bad_similarities = [self.base_model.compare(emb, prompt_embedding) for emb in self.bad_embs]

        avg_similarity = np.mean(similarities)
        # avg_bad_similarity = np.mean(bad_similarities)

        print(f"Average image similarity: {avg_similarity:.4f}")
        # print(f"Average bad image similarity: {avg_bad_similarity:.4f}")
        # print(f"Ratio (good/bad): {avg_good_similarity/avg_bad_similarity:.4f}")
        # print(f"Inverse ratio (bad/good): {avg_bad_similarity/avg_good_similarity:.4f}")

        return optimized_prompt


def evaluate_metric(metric_values, true_labels):
    """Evaluate how well a metric separates classes"""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    # Calculate separation (difference between means divided by sum of std)
    good_vals = metric_values[true_labels == 0]
    bad_vals = metric_values[true_labels == 1]
    
    # Find threshold that maximizes accuracy
    thresholds = sorted(metric_values)
    best_acc = 0
    best_thresh = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0

    for thresh in thresholds:
        pred_labels = (metric_values > thresh).astype(int)
        acc = accuracy_score(true_labels, pred_labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            try:
                best_precision = precision_score(true_labels, pred_labels)
                best_recall = recall_score(true_labels, pred_labels)
                best_f1 = f1_score(true_labels, pred_labels)
            except:
                best_precision = 0
                best_recall = 0
                best_f1 = 0

    # Calculate AUC
    try:
        auc = roc_auc_score(true_labels, metric_values)
    except:
        auc = 0

    return {
        "accuracy": best_acc,
        "threshold": best_thresh,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
        "auc": auc
    }


if __name__ == "__main__":
    print("Starting...")
    path = "/zhome/4a/b/137804/Desktop/autolbl"
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
            initial_prompt="[PAD] [PAD] [PAD]",
            indices=indices,
        )
        good_prompt = optimizer_good.optimize(
            maxiter=1000, optimizer="CMA"
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
        # good_prompt_emb = optimizer_good.base_model.embed_text(good_prompt)
        if vectorized:
            good_differences=[]
            good_files = [f for f in os.listdir(good_img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            good_files = [good_files[i] for i in indices]
            good_prompt_emb = torch.zeros((len(good_files), 512))
            for i in range(len(good_files)):
                torch.cuda.empty_cache()
                img_path = os.path.join(good_img_folder, good_files[i])
                img = PIL.Image.open(img_path)
                good_prompt=optimizer_good.optimize(
                    maxiter=1000, optimizer="CMA", img=img
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
            torch.cuda.empty_cache()
            img_path = os.path.join(good_img_folder, test_files[i])
            img = PIL.Image.open(img_path)
            emb = optimizer_good.base_model.embed_image(img)
            prompt_bad = optimizer_good.optimize(
                maxiter=100, optimizer="CMA", img=img
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

        # calculate LOO (leave one out) diffs for good images
        # Initialize arrays if they don't exist yet
        diff_LOO_max = np.array([])
        diff_LOO_min = np.array([])
        diff_LOO_mean = np.array([])
        diff_LOO_median = np.array([])
        
        for i, good_diff in enumerate(good_differences):
            torch.cuda.empty_cache()
            # Create a list of all other good differences excluding the current one
            other_good_diffs = [gd.cpu() for j, gd in enumerate(good_differences) if j != i]
            good_diffs_stacked = torch.stack(other_good_diffs)
            diffs = torch.abs(good_diff.cpu().unsqueeze(0) - good_diffs_stacked)
            diff_LLO_max = np.append(diff_LOO_max, torch.max(diffs).item())
            diff_LOO_min = np.append(diff_LOO_min, torch.min(diffs).item())
            diff_LOO_mean = np.append(diff_LOO_mean, torch.mean(diffs).item())
            diff_LOO_median = np.append(
                diff_LOO_median, torch.median(diffs).item()
            )
        metric_LOO=[diff_LOO_max, diff_LOO_min, diff_LOO_mean, diff_LOO_median]
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

        # After your existing loops, add this code:
        if vectorized:
            # Save raw differences for later analysis
            np.save(f"results/{bad_fldr}_raw_diffs.npy", {
                'test_files': test_files,
                'labels': labels,
                'bad_difference': bad_difference,  # already converted to numpy earlier
                'good_diffs': good_diffs_np,
            }, allow_pickle=True)

            # Calculate different metrics for comparison
            diff_test_max = np.zeros(len(test_files))
            diff_test_min = np.zeros(len(test_files))
            diff_test_mean = np.zeros(len(test_files))
            diff_test_median = np.zeros(len(test_files))

            # Recalculate metrics for each image
            for i in range(len(test_files)):
                # Use the already calculated bad_difference
                bd = torch.tensor(bad_difference[i])  # Convert back to tensor for easier calculation

                # Calculate differences with broadcasting
                diffs = torch.abs(bd.unsqueeze(0) - torch.tensor(good_diffs_np))

                # Calculate different metrics
                diff_test_max[i] = torch.max(diffs).item()
                diff_test_min[i] = torch.min(diffs).item()
                diff_test_mean[i] = torch.mean(diffs).item()
                diff_test_median[i] = torch.median(diffs).item()

            # Compare metrics with plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            # Plot each metric
            axes = [ax1, ax2, ax3, ax4]
            metrics = [diff_test_max, diff_test_min, diff_test_mean, diff_test_median]
            titles = ['Maximum', 'Minimum', 'Mean', 'Median']

            for ax, metric, title, mLOO in zip(axes, metrics, titles, metric_LOO):
                # Plot the data points
                ax.scatter(labels, metric, alpha=0.5)
                ax.scatter(np.ones(len(mLOO))*2, mLOO, label="good")
                ax.boxplot([metric[labels == 0], metric[labels == 1], mLOO], positions=[0, 1, 2])
                ax.set_ylabel(f"{title} Difference")
                ax.set_xlabel("True Label")
                ax.set_xticklabels(["Good Test", "Bad Test", "Good Train L0O"])

                # Calculate performance metrics
                metric_eval = evaluate_metric(metric, labels)
                best_acc = metric_eval['accuracy']
                threshold = metric_eval['threshold']
                f1_score = metric_eval['f1']

                # Add threshold line
                ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
                ax.legend()

                # Update title to show accuracy and F1 score
                ax.set_title(f"{title} - Acc: {best_acc:.4f}, F1: {f1_score:.4f}")

            plt.tight_layout()
            plt.savefig(f"results/{bad_fldr}_metric_comparison.png")

            # Create a summary table
            metrics_data = {
                'Max': evaluate_metric(diff_test_max, labels),
                'Min': evaluate_metric(diff_test_min, labels),
                'Mean': evaluate_metric(diff_test_mean, labels),
                'Median': evaluate_metric(diff_test_median, labels)
            }

            # Save metrics summary
            with open(f"results/{bad_fldr}_metrics_summary.txt", 'w') as f:
                f.write(f"Metrics comparison for {bad_fldr}\n")
                f.write("="*40 + "\n")
                for metric_name, results in metrics_data.items():
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
                    f.write(f"  Precision: {results['precision']:.4f}\n")
                    f.write(f"  Recall: {results['recall']:.4f}\n")
                    f.write(f"  F1 Score: {results['f1']:.4f}\n")
                    f.write(f"  AUC: {results['auc']:.4f}\n")
                    f.write(f"  Threshold: {results['threshold']:.4f}\n\n")
