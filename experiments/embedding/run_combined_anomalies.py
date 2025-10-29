import torch
import PIL
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import wandb

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


# Import the PromptOptimizer class from the experiments.embedding module
# Note: When running from the autolbl root directory, use this import
import sys
import os
# Add the autolbl root to path if running directly
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    autolbl_root = os.path.dirname(os.path.dirname(current_dir))
    if autolbl_root not in sys.path:
        sys.path.insert(0, autolbl_root)

from experiments.embedding.meta_emb_anomaly import PromptOptimizer

if __name__ == "__main__":
    wandb.init(project="meta_all", name="combined_anomalies")
    wandb.config.update({
        "model": "Meta-embedding",
        "dataset": "wood",
        "anomaly_types": ["color", "combined", "hole", "liquid", "scratch"],
        "vectorized": True,
        "optimizer": "differential_evolution",
        "maxiter": 1000
    })
    path = "D:/Data/dtu/OneDrive - Danmarks Tekniske Universitet/MSc MMC/5th semester/Thesis/autolbl"
    #check if path exists
    if not os.path.exists(path):
        path = "/zhome/4a/b/137804/Desktop/autolbl"
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Indices for training images
    vectorized = True
    anomaly_folders = ["color", "combined", "hole", "liquid", "scratch"]

    # Paths
    good_img_folder = os.path.join(path, "data/wood/test/good")

    # Create optimizer with good images only
    optimizer_good = PromptOptimizer(
        img_folder=good_img_folder,
        initial_prompt="wood",
        indices=indices,
    )
    good_prompt = optimizer_good.optimize(
        maxiter=1000, optimizer="differential_evolution"
    )

    # Prepare combined test data
    test_files = []
    labels = []
    source_folders = []  # Keep track of which folder each image comes from

    # Add good test files (not used in training)
    good_files = [
        f
        for f in os.listdir(good_img_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    for i, f in enumerate(good_files):
        if i not in indices:  # Skip training images
            test_files.append(os.path.join(good_img_folder, f))
            labels.append(0)  # 0 = good
            source_folders.append("good")

    # Add all anomaly files
    for anomaly_type in anomaly_folders:
        bad_img_folder = os.path.join(path, f"data/wood/test/{anomaly_type}")
        if os.path.exists(bad_img_folder):
            bad_files = [
                f
                for f in os.listdir(bad_img_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            for f in bad_files:
                test_files.append(os.path.join(bad_img_folder, f))
                labels.append(1)  # 1 = anomaly
                source_folders.append(anomaly_type)

    # Convert to numpy arrays
    labels = np.array(labels)

    # Prepare good differences (same as before)
    if vectorized:
        good_differences = []
        good_files = [
            f
            for f in os.listdir(good_img_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        good_files = [good_files[i] for i in indices]
        good_prompt_emb = torch.zeros((len(good_files), 512))

        for i in range(len(good_files)):
            torch.cuda.empty_cache()
            img_path = os.path.join(good_img_folder, good_files[i])
            img = PIL.Image.open(img_path)
            good_prompt = optimizer_good.optimize(
                maxiter=1000, optimizer="differential_evolution", img=img
            )
            good_prompt_emb[i] = optimizer_good.base_model.embed_text(good_prompt)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            good_differences.append(
                optimizer_good.base_model.vec_difference(
                    optimizer_good.embs[i].to(device), good_prompt_emb[i].to(device)
                ).cpu()
            )

    # Process all test images - both good and anomalies
    diff_test = np.zeros(len(test_files))
    if vectorized:
        bad_difference = torch.zeros((len(test_files), 512))
    else:
        bad_difference = np.zeros(len(test_files))

    for i in tqdm(range(len(test_files)), desc="Processing test images"):
        torch.cuda.empty_cache()
        img_path = test_files[i]
        img = PIL.Image.open(img_path)
        emb = optimizer_good.base_model.embed_image(img)
        prompt_bad = optimizer_good.optimize(
            maxiter=1000, optimizer="differential_evolution", img=img
        )

        if vectorized:
            bad_difference[i] = optimizer_good.base_model.vec_difference(
                emb, optimizer_good.base_model.embed_text(prompt_bad)
            ).cpu()

            # Calculate differences with broadcasting
            good_diffs_stacked = torch.stack([gd.cpu() for gd in good_differences])
            diffs = torch.abs(bad_difference[i].unsqueeze(0) - good_diffs_stacked)
            diff_test[i] = torch.median(diffs).item()
        else:
            bad_difference[i] = optimizer_good.base_model.difference(
                emb, optimizer_good.base_model.embed_text(prompt_bad)
            ).cpu()
            diff_test[i] = np.max(
                np.abs(bad_difference[i] - np.array(good_differences)), axis=0
            )
        torch.cuda.empty_cache()

    # Calculate LOO metrics (same as before)
    diff_LOO_max = []
    diff_LOO_min = []
    diff_LOO_mean = []
    diff_LOO_median = []

    for i, good_diff in enumerate(good_differences):
        torch.cuda.empty_cache()
        other_good_diffs = [gd.cpu() for j, gd in enumerate(good_differences) if j != i]
        good_diffs_stacked = torch.stack(other_good_diffs)
        diffs = torch.abs(good_diff.cpu().unsqueeze(0) - good_diffs_stacked)
        diff_LOO_max.append(torch.max(diffs).item())
        diff_LOO_min.append(torch.min(diffs).item())
        diff_LOO_mean.append(torch.mean(diffs).item())
        diff_LOO_median.append(torch.median(diffs).item())

    metric_LOO = [diff_LOO_max, diff_LOO_min, diff_LOO_mean, diff_LOO_median]

    # Save raw differences for later analysis
    if vectorized:
        good_diffs_np = torch.stack([gd.cpu() for gd in good_differences]).cpu().numpy()
        bad_difference_np = bad_difference.cpu().numpy()

        # Save combined result
        np.save(
            "results/combined_anomalies_raw_diffs.npy",
            {
                "test_files": test_files,
                "labels": labels,
                "source_folders": source_folders,  # Added to track which folder each image comes from
                "bad_difference": bad_difference_np,
                "good_diffs": good_diffs_np,
            },
            allow_pickle=True,
        )

        # Calculate metrics for evaluation
        diff_test_max = np.zeros(len(test_files))
        diff_test_min = np.zeros(len(test_files))
        diff_test_mean = np.zeros(len(test_files))
        diff_test_median = np.zeros(len(test_files))

        for i in range(len(test_files)):
            bd = torch.tensor(bad_difference_np[i])
            diffs = torch.abs(bd.unsqueeze(0) - torch.tensor(good_diffs_np))
            diff_test_max[i] = torch.max(diffs).item()
            diff_test_min[i] = torch.min(diffs).item()
            diff_test_mean[i] = torch.mean(diffs).item()
            diff_test_median[i] = torch.median(diffs).item()

        # Create visualization and metrics report
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        axes = [ax1, ax2, ax3, ax4]
        metrics = [diff_test_max, diff_test_min, diff_test_mean, diff_test_median]
        titles = ["Maximum", "Minimum", "Mean", "Median"]

        for ax, metric, title, mLOO in zip(axes, metrics, titles, metric_LOO):
            ax.scatter(labels, metric, alpha=0.5)
            ax.scatter(np.ones(len(mLOO)) * 2, mLOO, label="good")
            ax.boxplot(
                [metric[labels == 0], metric[labels == 1], mLOO],
                positions=[0, 1, 2],
            )
            ax.set_ylabel(f"{title} Difference")
            ax.set_xlabel("True Label")
            ax.set_xticklabels(["Good Test", "Anomaly Test", "Good Train LLO"])

            # Calculate performance metrics
            metric_eval = evaluate_metric(metric, labels)
            best_acc = metric_eval["accuracy"]
            threshold = metric_eval["threshold"]
            f1_score_val = metric_eval["f1"]

            ax.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                label=f"Threshold: {threshold:.4f}",
            )
            ax.legend()
            ax.set_title(f"{title} - Acc: {best_acc:.4f}, F1: {f1_score_val:.4f}")

        plt.tight_layout()
        plt.savefig("results/combined_anomalies_metric_comparison.png")

        # Create summary table
        metrics_data = {
            "Max": evaluate_metric(diff_test_max, labels),
            "Min": evaluate_metric(diff_test_min, labels),
            "Mean": evaluate_metric(diff_test_mean, labels),
            "Median": evaluate_metric(diff_test_median, labels),
        }

        # Save metrics summary
        with open("results/combined_anomalies_metrics_summary.txt", "w") as f:
            f.write("Metrics comparison for combined anomalies\n")
            f.write("=" * 50 + "\n")
            for metric_name, results in metrics_data.items():
                f.write(f"{metric_name}:\n")
                f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"  Precision: {results['precision']:.4f}\n")
                f.write(f"  Recall: {results['recall']:.4f}\n")
                f.write(f"  F1 Score: {results['f1']:.4f}\n")
                f.write(f"  AUC: {results['auc']:.4f}\n")
                f.write(f"  Threshold: {results['threshold']:.4f}\n\n")

        print("Processing complete! Results saved to 'results/combined_anomalies_*'")
