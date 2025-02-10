import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import wandb
from autodistill.utils import plot
import time
import pandas as pd
def compare_plot(dataset, gt_dataset, results_dir="results"):
    wandb_image_tab = wandb.Table(columns=["Image_ID", "GT_Annotation", "Inference_Annotation"])
    wandb.log({"Comparison Images": wandb_image_tab})
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
    wandb_images = []
    wandb_gt_images = []
    #time how long to make gt_dict
    start_time = time.time()
    gt_dict = {os.path.splitext(os.path.basename(image_path))[0] + ".jpg": (image_path, annotation) for image_path, _, annotation in gt_dataset}
    print(f"Time to make gt_dict: {time.time()-start_time}")
    wandb.log({"Time to make gt_dict": time.time()-start_time})
    # Process dataset images and ground truth images together
    for image_path, _, annotation in dataset:
        image = cv2.imread(image_path)
        classes = dataset.classes
        result = annotation

        wandb_img = detections_to_wandb(image, result, classes)
        #log wandb image  
        wandb_images.append(wandb_img)
        #add to wandb table
        wandb_image_tab.add_data(os.path.basename(image_path), wandb_img, None)
        try:
            img.append(plot(image=image, classes=classes, detections=result, raw=True))
        except Exception as e:
            print(f"Error plotting inference image: {e}")
            img.append(plot(image=image, classes=[str(i) for i in range(100)], detections=result, raw=True))
        name.append(os.path.basename(image_path))

        name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        #wandb.log({f"inference_{name_gt}": wandb_img})
        if name_gt in gt_dict:
            gt_image_path, gt_annotation = gt_dict[name_gt]
            gt_classes = gt_dataset.classes
            gt_image = cv2.imread(gt_image_path)
            gt_result = gt_annotation
            wandb_gt_img = detections_to_wandb(gt_image, gt_result, gt_classes)
            #wandb.log({f"gt_{name_gt}": wandb_gt_img})
            wandb_gt_images.append(wandb_gt_img)
            if len(gt_result) == 0:
                img_gt = gt_image
            else:
                try:
                    if gt_result.confidence is None:
                        gt_result.confidence = np.ones_like(gt_result.class_id)
                    #img_gt = plot(image=gt_image, classes=gt_classes, detections=gt_result, raw=True)
                except Exception as e:
                    print(f"Error plotting ground truth image: {e}")
                    #img_gt = plot(image=gt_image, classes=[str(i) for i in range(100)], detections=gt_result, raw=True)

            # Find fig index
            """
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
            """
        #wandb_image_tab.add_data(name, wandb_gt_images, wandb_images)
        wandb_image_tab.add_data(name_gt, wandb_gt_img, wandb_img)

        #update_table_wandb("Comparison Images", [name_gt, wandb_gt_img, wandb_img])
        wandb.log({"Comparison Images": wandb_image_tab})
    """
    try:
        update_table_wandb("Comparison Images", [name_gt, wandb_gt_img, wandb_img])
    except Exception as e:
        print(f"Error updating table: {e}")
    """
    wandb.log({"Comparison Images": wandb_image_tab})
    df = pd.DataFrame({"Image_ID": name, "GT_Annotation": wandb_gt_images, "Inference_Annotation": wandb_images})
    wandb_tab2 = wandb.Table(dataframe=df, allow_mixed_types=True)
    wandb.log({"Comparison Images2": wandb_tab2})
    
def update_table_wandb(table_name, row, run_id = None):
    # Ensure the table_name is in the correct format 'collection:alias'
    if run_id == None:
        run_id = wandb.run.id
    #remove spece from table name
    table_name = table_name.replace(" ", "")
    table_tag = f"run-{run_id}-{table_name}:latest"
    table = wandb.use_artifact(table_tag).get(table_name)
    table.add_data(*row)
    #get column names
    columns = table.columns
    # Reinitialize the table with its updated data to ensure compatibility
    updated_table = wandb.Table(data=table.data, columns=columns, allow_mixed_types=True)
    # Log the updated table to Weights & Biases
    wandb.log({table_name: updated_table})

def detections_to_wandb(img, detections, classes)->wandb.Image:
    """
    Convert attention location to W&B image with bounding boxes.
    Args:
        img (PIL.Image): The input image.
        attn (np.ndarray): The attention location.
    Returns:
        wandb.Image: The W&B image.
    """
    class_labels = {i: classes[i] for i in range(len(classes))}
    boxes = {"predictions": {"box_data": [], "class_labels": class_labels}}
    for detection in detections:
        bbox = detection[0]
        conf = detection[2] if detection[2] is not None else 2.0  # Set default confidence if None
        class_id = int(detection[3])
        #print(f"Class: {class_id}")
        #print(f"Confidence: {conf}")
        # Check if bbox has no 0 values
        if bbox.any() != 0:
            x1, y1, x2, y2 = bbox
            #turn all to float
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            #print(f"Box: {x1}, {y1}, {x2}, {y2}")

            boxes["predictions"]["box_data"].append({
                "position": {
                    "middle": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                },
                "domain": "pixel",
                "class_id": class_id,
                "box_caption": f"{classes[class_id]}",
                "scores": {
                    "confidence": float(conf)
                }
            })
    return wandb.Image(img, boxes=boxes)