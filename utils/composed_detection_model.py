import numpy as np
import os
import glob
import cv2

import roboflow
from tqdm import tqdm
import supervision as sv
from PIL import Image
import enum
from autodistill.detection.detection_base_model import DetectionBaseModel
from autodistill.helpers import load_image, split_data

DEFAULT_LABEL_ANNOTATOR = sv.LabelAnnotator(text_position=sv.Position.CENTER)
SET_OF_MARKS_SUPPORTED_MODELS = ["GPT4V"]
class NmsSetting(str, enum.Enum):
    NONE = "no_nms"
    CLASS_SPECIFIC = "class_specific"
    CLASS_AGNOSTIC = "class_agnostic"


class ComposedDetectionModel2(DetectionBaseModel):
    """
    Run inference with a detection model then run inference with a classification model on the detected regions.
    """

    def __init__(
        self,
        detection_model,
        classification_model,
        set_of_marks=None,
        set_of_marks_annotator=DEFAULT_LABEL_ANNOTATOR,
    ):
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.set_of_marks = set_of_marks
        self.set_of_marks_annotator = set_of_marks_annotator
        self.ontology = self.classification_model.ontology

    def predict(self, image: str) -> sv.Detections:
        """
        Run inference with a detection model then run inference with a classification model on the detected regions.

        Args:
            image: The image to run inference on
            annotator: The annotator to use to annotate the image

        Returns:
            detections (sv.Detections)
        """
        opened_image = Image.open(image)

        detections = self.detection_model.predict(image)

        if self.set_of_marks is not None:
            labels = [f"{num}" for num in range(len(detections.xyxy))]

            opened_image = np.array(opened_image)

            annotated_frame = self.set_of_marks_annotator.annotate(
                scene=opened_image, labels=labels, detections=detections
            )

            opened_image = Image.fromarray(annotated_frame)

            opened_image.save("temp.jpeg")

            if not hasattr(self.classification_model, "set_of_marks"):
                raise Exception(
                    f"The set classification model does not have a set_of_marks method. Supported models: {SET_OF_MARKS_SUPPORTED_MODELS}"
                )

            result = self.classification_model.set_of_marks(
                input=image, masked_input="temp.jpeg", classes=labels, masks=detections
            )

            return detections

        for pred_idx, bbox in enumerate(detections.xyxy):
            # extract region from image
            region = opened_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            # save as tempfile
            region.save("temp.jpeg")

            result = self.classification_model.predict("temp.jpeg")

            if len(result.class_id) == 0:
                continue

            result = result.get_top_k(1)[0][0]

            detections.class_id[pred_idx] = result

        return detections
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
    ) -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
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
                #detections = self.predict(image)
                detections = self.predict(f_path)

            if nms_settings == NmsSetting.CLASS_SPECIFIC:
                detections = detections.with_nms()
            if nms_settings == NmsSetting.CLASS_AGNOSTIC:
                detections = detections.with_nms(class_agnostic=True)

            detections_map[f_path] = detections

        dataset = sv.DetectionDataset(
            self.ontology.classes(), image_paths, detections_map
        )

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
        split_data(output_folder, record_confidence=record_confidence)

        if human_in_the_loop:
            roboflow.login()

            rf = roboflow.Roboflow()

            workspace = rf.workspace()

            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        print("Labeled dataset created - ready for distillation.")
        return dataset
