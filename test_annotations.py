# Example usage
import supervision as sv
class Detections:
    def __init__(self, data):
        self.data = data

def approximate_mask_with_polygons(mask, min_image_area_percentage, max_image_area_percentage, approximation_percentage):
    # Dummy implementation
    return [mask]

def polygon_to_xyxy(polygon):
    # Dummy implementation
    return polygon

def object_to_yolo(xyxy, class_id, image_shape, polygon=None):
    # Dummy implementation
    return f"{class_id} " + " ".join(map(str, xyxy))

detections = Detections(
    [
        ([0.71321, 0.00000, 0.71321, 0.00293], None, None, 0, None),
        ([0.71286, 0.00000, 0.71286, 0.00293], None, None, 0, None),
        # Add more detections as needed
    ]
)
image_shape = (1024, 768, 3)
annotations = sv.DetectionDataset.detections_to_yolo_annotations(
    detections, image_shape
)
for annotation in annotations:
    print(annotation)
