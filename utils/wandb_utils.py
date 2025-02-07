import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import wandb

def compare_plot(dataset, gt_dataset, results_dir="results"):
    wandb.Table(columns=["Image_ID", "GT_Annotation", "Inference_Annotation"])
    # Ensure confidence is set for all annotations in both datasets
    for key in dataset.annotations.keys():
        for i in range(len(dataset.annotations[key])):
            dataset.annotations[key][i].confidence = np.ones_like(
                dataset.annotations[key][i].class_id
            )
    for key in gt_dataset.annotations.keys():
        for i in range(len(gt_dataset.annotations[key])):
            gt_dataset.annotations[key][i].confidence = np.ones_like(
                gt_dataset.annotations[key][i].class_id
            )

    img = []
    name = []

    # Process dataset images
    for image_path, _, annotation in dataset:
        image = cv2.imread(image_path)
        classes = dataset.classes
        result = annotation
        try:
            img.append(plot(image=image, classes=classes, detections=result, raw=True))
        except Exception as e:
            print(f"Error plotting inference image: {e}")
            img.append(plot(image=image, classes=[str(i) for i in range(100)], detections=result, raw=True))
        name.append(os.path.basename(image_path))

        
    # Process ground truth images
    for image_path, _, annotation in gt_dataset:
        classes = gt_dataset.classes
        name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        if name_gt in name:
            image = cv2.imread(image_path)
            result = annotation
            if len(result) == 0:
                img_gt = image
            else:
                try:
                    if result.confidence is None:
                        result.confidence = np.ones_like(result.class_id)
                    img_gt = plot(image=image, classes=classes, detections=result, raw=True)
                except Exception as e:
                    print(f"Error plotting ground truth image: {e}")
                    img_gt = plot(image=image, classes=[str(i) for i in range(100)], detections=result, raw=True)

            # Find fig index
            index = name.index(name_gt)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
            axes[0].imshow(img[index])
            axes[0].set_title("Inference")
            axes[0].axis("off")
            axes[1].imshow(img_gt)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            fig.patch.set_facecolor('none')

            try:
                wandb.log({f"Annotated Image {name_gt}": wandb.Image(fig)})
            except Exception as e:
                print(f"WandB logging error: {e}")
            plt.savefig(os.path.join(results_dir, name_gt), dpi=1200)
            plt.close(fig)