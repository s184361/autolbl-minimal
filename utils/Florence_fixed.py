import os
from dataclasses import dataclass
import enum
import wandb
import glob
import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import (CaptionOntology, DetectionBaseModel,
                                   DetectionTargetModel)
from autodistill.helpers import load_image
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)
from autodistill.helpers import load_image, split_data
from utils.embedding_ontology import EmbeddingOntologyImage
from utils.check_labels import evaluate_detections,log_evaluation_results

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NmsSetting(str, enum.Enum):
    NONE = "no_nms"
    CLASS_SPECIFIC = "class_specific"
    CLASS_AGNOSTIC = "class_agnostic"

class DetectionsDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset):
        self.dataset = dataset
        self.keys = list(dataset.images.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        keys = list(self.dataset.images.keys())
        key = self.keys[idx]
        image = self.dataset.images[key]
        annotations = self.dataset.annotations[key]
        h, w, _ = image.shape

        boxes = (annotations.xyxy / np.array([w, h, w, h]) * 1000).astype(int).tolist()
        labels = [self.dataset.classes[idx] for idx in annotations.class_id]

        prefix = "<OD>"

        suffix_components = []
        for [x1, y1, x2, y2], label in zip(boxes, labels):
            suffix_component = f"{label}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
            suffix_components.append(suffix_component)

        suffix = "".join(suffix_components)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return prefix, suffix, image


def run_example(task_prompt, processor, model, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )

    return parsed_answer


@dataclass
class Florence2(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        model_id = "microsoft/Florence-2-large"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, device_map="cuda"
        )
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = load_image(input, return_format="PIL")
        ontology_classes = self.ontology.classes()
        ontology_prompts = self.ontology.prompts()
        PROMPT ="and ".join(ontology_prompts) + "." #"A photo of " + ", and ".join(ontology_prompts) + "."
        result = run_example(
            "<CAPTION_TO_PHRASE_GROUNDING>",
            self.processor,
            self.model,
            image,
            PROMPT,
        )
        results = result["<CAPTION_TO_PHRASE_GROUNDING>"]
        boxes_and_labels = list(zip(results["bboxes"], results["labels"]))
        
        # Split the ontology_prompts into individual labels
        ontology_labels = [label.strip() for label in ontology_prompts[0].split('.') if label.strip()]

        valid_detections = [
            box
            for box, label in boxes_and_labels
            if label in ontology_labels
        ]

        if len(valid_detections) == 0 and len(ontology_classes) > 1:
            print("No detections found or too many classes detected")
            return sv.Detections.empty()
        
        h, w = image.size
        
        # Filter boxes covering more than 95% of the image area
        filtered_boxes = []
        filtered_class_ids = []
        filtered_confidences = []
        
        for box, label in boxes_and_labels:
            # Correctly calculate box area: (x2-x1) * (y2-y1)
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            image_area = h * w
            
            if box_area < 0.95 * image_area:
                filtered_boxes.append(box)
                filtered_class_ids.append(0)  # Using 0 as default class ID
                filtered_confidences.append(1.0)

        # Handle case when all boxes are filtered out
        if not filtered_boxes:
            return sv.Detections.empty()
            
        detections = sv.Detections(
            xyxy=np.array(filtered_boxes),
            class_id=np.array(filtered_class_ids),
            confidence=np.array(filtered_confidences),
        )

        detections = detections[detections.confidence > confidence]
        return detections


class Florence2Trainer(DetectionTargetModel):
    def __init__(
        self,
        checkpoint: str = "microsoft/Florence-2-base-ft",
    ):
        REVISION = "refs/pr/6"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True, revision=REVISION
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, revision=REVISION
        )

        self.model = model
        self.processor = processor
        self.REVISION = REVISION

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        image = Image.open(input)
        task = "<OD>"
        text = "<OD>"

        inputs = self.processor(text=text, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = self.peft_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        response = self.processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        detections = sv.Detections.from_lmm(
            sv.LMM.FLORENCE_2, response, resolution_wh=image.size
        )
        detections = detections[detections.confidence > confidence]

        return detections

    def train(self, dataset_path, epochs=10):
        ds_train = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset_path}/train",
            annotations_path=f"{dataset_path}/train/_annotations.coco.json",
        )

        ds_valid = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset_path}/valid",
            annotations_path=f"{dataset_path}/valid/_annotations.coco.json",
        )

        BATCH_SIZE = 6
        NUM_WORKERS = 0

        def collate_fn(batch):
            questions, answers, images = zip(*batch)
            inputs = self.processor(
                text=list(questions),
                images=list(images),
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)
            return inputs, answers

        train_dataset = DetectionsDataset(ds_train)
        val_dataset = DetectionsDataset(ds_valid)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
        )

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "linear",
                "Conv2d",
                "lm_head",
                "fc2",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
            revision=self.REVISION,
        )

        peft_model = get_peft_model(self.model, config)
        peft_model.print_trainable_parameters()
        self.peft_model = peft_model

        torch.cuda.empty_cache()

        EPOCHS = 10
        LR = 5e-6

        optimizer = AdamW(self.model.parameters(), lr=LR)
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for epoch in range(EPOCHS):
            self.model.train()
            train_loss = 0
            for inputs, answers in tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
            ):

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = self.processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(DEVICE)

                outputs = self.model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                loss.backward(), optimizer.step(), lr_scheduler.step(), optimizer.zero_grad()
                train_loss += loss.item()
                
                # After optimization, decode the prompt to see what it's learning
                with torch.no_grad():
                    # Extract images from inputs for prediction
                    # Get raw images from batch for prediction
                    actual_batch_size = inputs["pixel_values"].shape[0]
                    for i in range(actual_batch_size):
                        # Convert processed pixel_values back to images for evaluation
                        # This is a simplified approach - ideally you'd track original images
                        img_tensor = inputs["pixel_values"][i].cpu().numpy()
                        # Process single image for prediction
                        if i == 0:  # Just evaluate the first image for efficiency
                            # Convert from tensor to PIL for prediction
                            img_tensor = np.transpose(img_tensor, (1, 2, 0))
                            # Normalize to 0-255 range
                            img_tensor = (img_tensor * 255).astype(np.uint8)
                            img_pil = Image.fromarray(img_tensor)
                            # Get detections for this single image
                            detections = self.predict(img_pil)
                            # Evaluate the detections
                            try:
                                #evaluate the detections
                                confusion_matrix, precision, recall, F1, map_result = evaluate_detections(detections, val_dataset)
                                # Log the evaluation metrics to wandb
                                log_evaluation_results(
                                    confusion_matrix, precision, recall, F1, map_result
                                )
                            except Exception as e:
                                print(f"Error evaluating detections: {e}")

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss}")

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, answers in tqdm(
                    val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
                ):

                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]
                    labels = self.processor.tokenizer(
                        text=answers,
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False,
                    ).input_ids.to(DEVICE)

                    outputs = self.model(
                        input_ids=input_ids, pixel_values=pixel_values, labels=labels
                    )
                    loss = outputs.loss

                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                print(f"Average Validation Loss: {avg_val_loss}")

            output_dir = f"./model_checkpoints/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.processor.save_pretrained(output_dir)


class Florence2Prompt(DetectionTargetModel):
    def __init__(
        self,
        checkpoint: str = "microsoft/Florence-2-base-ft",
        initial_prompt: str = "A photo of",
    ):
        REVISION = "refs/pr/6"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor but don't modify the model weights
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True, revision=REVISION
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, revision=REVISION
        )

        # Freeze all model parameters to ensure we only optimize the prompt
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.processor = processor
        self.REVISION = REVISION
        self.DEVICE = DEVICE
        self.initial_prompt = initial_prompt
        self.optimized_prompt = initial_prompt  # Will be updated during training
        self.peft_model = model  # For compatibility with predict method
        self.ontology = None
    def get_embeddings(self, text):
        """Get token embeddings for text"""
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True
        ).to(self.DEVICE)
        return self.model.get_input_embeddings()(inputs["input_ids"])
        
    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        if isinstance(input, str):
            image = Image.open(input)
        elif isinstance(input, Image.Image):
            image = input
        elif isinstance(input, np.ndarray):
            image = Image.fromarray(input)
        else:
            #print(input)
            image = input
        
        # Use the optimized prompt if available
        #task = "<OD>"
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        text = self.optimized_prompt

        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.DEVICE)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        response = self.processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        detections = sv.Detections.from_lmm(
            sv.LMM.FLORENCE_2, response, resolution_wh=image.size
        )
        detections = detections[detections.confidence > confidence]

        return detections

    def train(self, ds_train=None, ds_valid=None, epochs=10, lr=1e-2, prompt_len=10):
        if ds_train is None or ds_valid is None:
            raise ValueError("Please provide training and validation datasets")
        BATCH_SIZE = 30
        NUM_WORKERS = 0

        # Create learnable prompt embeddings
        vocab_size = self.processor.tokenizer.vocab_size
        tokenized_prompt = self.processor.tokenizer(
            self.initial_prompt, return_tensors="pt", padding=True
        ).input_ids.to(self.DEVICE)
        
        # Extract initial embeddings for the prompt
        initial_prompt_embeds = self.model.get_input_embeddings()(tokenized_prompt).detach()
        
        # Get embedding dimensions
        embed_dim = initial_prompt_embeds.shape[-1]
        seq_len = initial_prompt_embeds.shape[1]
        
        # Create optimizable prompt embeddings - one per batch element
        prompt_embeds = torch.nn.Parameter(
            initial_prompt_embeds, 
            requires_grad=True
        )
        print(f"Initial prompt embeddings shape: {prompt_embeds.shape}")
        # Create optimizer for prompt embeddings
        optimizer = AdamW([prompt_embeds], lr=lr)

        def collate_fn(batch):
            questions, answers, images = zip(*batch)
            # Process images but not questions (we'll use our learnable prompt)
            inputs = self.processor(
                images=list(images),
                return_tensors="pt",
                padding=True,
            ).to(self.DEVICE)
            return inputs, answers

        train_dataset = DetectionsDataset(ds_train)
        val_dataset = DetectionsDataset(ds_valid)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
        )

        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        best_val_loss = float('inf')
        best_prompt_embeds = prompt_embeds.clone().detach()

        for epoch in range(epochs):
            self.model.eval()  # Keep model in eval mode since we're not training it
            train_loss = 0
            
            for inputs, answers in tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
            ):
                # Get actual batch size (may be smaller than BATCH_SIZE for last batch)
                actual_batch_size = inputs["pixel_values"].shape[0]
                
                # Use repeat the prompt embeddings for this batch
                batch_embeds = prompt_embeds.repeat(actual_batch_size, 1, 1)
                
                # Use our optimizable prompt embeddings instead of input_ids
                pixel_values = inputs["pixel_values"]
                
                # Prepare target labels
                labels = self.processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(self.DEVICE)
                
                # Create attention mask (1s for all tokens in the prompt)
                attention_mask = torch.ones((actual_batch_size, seq_len), device=self.DEVICE)

                # Forward pass with prompt embeddings and attention mask
                outputs = self.model(
                    inputs_embeds=batch_embeds,  # Use learned prompt embeddings
                    pixel_values=pixel_values,
                    labels=labels,
                    attention_mask=attention_mask
                )
                loss = outputs.loss
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                train_loss += loss.item()
                
                # After optimization, decode the prompt to see what it's learning
                with torch.no_grad():
                    image_paths = sv.list_files_with_extensions(
                        directory=os.path.dirname(ds_valid.image_paths[0]),
                        extensions=["jpg", "png"]
                    )
                    progress_bar = tqdm(image_paths, desc="Labeling images")
                    detections_map = {}
                    for f_path in progress_bar:
                        progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)

                        image = cv2.imread(f_path)
                        detections = self.predict(image)
                        detections_map[f_path] = detections
                    dataset = sv.DetectionDataset(
                        ["defect"], image_paths, detections_map
                    )
                    try:
                        #check if the gt_dataset is correct
                        confusion_matrix, precision, recall, F1, map50, map50_95=evaluate_detections(dataset, val_dataset)
                        # Log the evaluation metrics to wandb
                        print(f"Precision: {precision}, Recall: {recall}, F1: {F1}, mAP50: {map50}, mAP50-95: {map50_95}")
                        log_evaluation_results(
                            confusion_matrix, precision, recall, F1, map50, map50_95
                        )
                    except Exception as e:
                        print(f"Error evaluating detections: {e}")
                            # Try to decode prompt embeddings to see current prompt
                    # This is approximate as direct inversion isn't always perfect
                    logits = self.model.get_input_embeddings().weight.mm(prompt_embeds[0].T)
                    token_ids = torch.argmax(logits, dim=0)
                    current_prompt = self.processor.tokenizer.decode(token_ids)
                
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss}")
            print(f"Current optimized prompt: {current_prompt}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, answers in tqdm(
                    val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
                ):
                    # Get actual batch size
                    actual_batch_size = inputs["pixel_values"].shape[0]
                    
                    # Use prompt embeddings for this batch  
                    batch_embeds = prompt_embeds.repeat(actual_batch_size, 1, 1)
                    
                    pixel_values = inputs["pixel_values"]
                    labels = self.processor.tokenizer(
                        text=answers,
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False,
                    ).input_ids.to(self.DEVICE)

                    # Create attention mask (1s for all tokens in the prompt)
                    attention_mask = torch.ones((actual_batch_size, seq_len), device=self.DEVICE)

                    # Forward pass with learned prompt
                    outputs = self.model(
                        inputs_embeds=batch_embeds,
                        pixel_values=pixel_values,
                        labels=labels,
                        attention_mask=attention_mask
                    )
                    loss = outputs.loss
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                print(f"Validation Loss: {avg_val_loss}")
                
                try:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                            "token_ids": token_ids.tolist(),
                        }
                    )
                except:
                    pass
                # Save best prompt
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_prompt_embeds = prompt_embeds.clone().detach()
                    print("New best prompt found!")
        
        # Save the best prompt embeddings
        with torch.no_grad():
            logits = self.model.get_input_embeddings().weight.mm(best_prompt_embeds[0].T)
            token_ids = torch.argmax(logits, dim=0)
            final_prompt = self.processor.tokenizer.decode(token_ids)
            
        print(f"Final optimized prompt: {final_prompt}")
        self.optimized_prompt = final_prompt
        
        # Save prompt to file
        os.makedirs("./optimized_prompts", exist_ok=True)
        with open("./optimized_prompts/best_prompt.txt", "w") as f:
            f.write(final_prompt)
            
        return final_prompt
    
    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        output_folder: str | None = None,
        human_in_the_loop: bool = False,
        roboflow_project: str | None = None,
        roboflow_tags: list[str] = ["autodistill"],
        sahi: bool = False,
        record_confidence: bool = False,
        nms_settings: NmsSetting = NmsSetting.NONE,
        save_images=True
    ) -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        if self.ontology is None:
            self.ontology = CaptionOntology({self.optimized_prompt: "defect"})

        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        image_paths = glob.glob(input_folder + "/*" + extension)
        detections_map = {}

        if sahi:
            slicer = sv.InferenceSlicer(callback=self.predict)

        progress_bar = tqdm(image_paths, desc="Labeling images")
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)

            image = cv2.imread(f_path)
            if sahi:
                detections = slicer(image)
            else:
                detections = self.predict(image)

            if nms_settings == NmsSetting.CLASS_SPECIFIC:
                detections = detections.with_nms()
            if nms_settings == NmsSetting.CLASS_AGNOSTIC:
                detections = detections.with_nms(class_agnostic=True)

            detections_map[f_path] = detections

        if isinstance(self.ontology, EmbeddingOntologyImage):
            dataset = sv.DetectionDataset(
                self.ontology.prompts(), image_paths, detections_map
            )
        else:
            dataset = sv.DetectionDataset(
                self.ontology.classes(), image_paths, detections_map
            )
        print(dataset)
        dataset.as_yolo(
            output_folder + "/images",
            output_folder + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=output_folder + "/data.yaml",
        )

        if record_confidence:
            image_names = [os.path.basename(f_path) for f_path in image_paths]
            self._record_confidence_in_files(
                output_folder + "/annotations", image_names, detections_map
            )
        if save_images:
            print("Copying images to output folder...")
            split_data(output_folder, record_confidence=record_confidence)

        print("Labeled dataset created - ready for distillation.")
        return dataset