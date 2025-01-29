from typing import Any, Optional, Dict
from pydantic import BaseModel, Field
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image, ExifTags
from collections import Counter
import cv2
import os
import datetime
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt


class ImageProcessor(BaseModel):
    """
    A utility class for processing a specific image, including object detection,
    caption generation, datetime extraction, and visualization.
    """
    image_path: str = Field(..., description="Path to the image to process")
    
    _blip_processor: Optional[Blip2Processor] = None
    _blip_model: Optional[Blip2ForConditionalGeneration] = None
    _detectron_cfg: Optional[Any] = None
    _predictor: Optional[DefaultPredictor] = None
    _class_names: Optional[list] = None
    _metadata: Optional[Dict[str, str]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_shared_resources()

    @classmethod
    def _initialize_shared_resources(cls):
        """Initialize shared models and configurations if not already initialized."""
        cls._initialize_blip()
        cls._initialize_detectron()

    @classmethod
    def _initialize_blip(cls):
        """Initialize the BLIP-2 model."""
        try:
            cls._blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            cls._blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
            cls._blip_initialized = True
            print("BLIP models initialized.")
        except Exception as e:
            raise RuntimeError(f"Error initializing BLIP models (BLIP-2): {e}")

    @classmethod
    def _initialize_detectron(cls):
        """Initialize the Detectron2 model and configuration."""
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cpu"
            cls._detectron_cfg = cfg
            cls._predictor = DefaultPredictor(cfg)
            print("Detectron2 model initialized.")
            cls._metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            cls._class_names = cls._metadata.get("thing_classes", [])
            cls._detectron_initialized = True
            print("Detectron2 model initialized.")
        except Exception as e:
            raise RuntimeError(f"Error initializing Detectron2: {e}")

    def detect_objects(self) -> Counter[str]:
        """Detect objects in the image using Detectron2."""
        if self._predictor is None:
            raise RuntimeError("Detectron2 model is not initialized.")

        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not read image at {self.image_path}")

        outputs = self._predictor(image)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.numpy()
        detected_items = [self._class_names[c] for c in pred_classes]
        return Counter(detected_items)

    def visualize_image(self):
        """Visualize the image with object detection annotations."""
        if self._predictor is None:
            raise RuntimeError("Detectron2 model is not initialized.")

        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not read image at {self.image_path}")

        outputs = self._predictor(image)
        v = Visualizer(image[:, :, ::-1], self._metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(10, 10))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis("off")
        plt.show()

    def describe_image(self) -> str:
        """Generate a caption/description for the image using BLIP-2."""
        if self._blip_processor is None or self._blip_model is None:
            raise RuntimeError("BLIP models are not initialized.")

        try:
            image = Image.open(self.image_path).convert("RGB")
            inputs = self._blip_processor(image, return_tensors="pt")
            outputs = self._blip_model.generate(**inputs)
            return self._blip_processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise ValueError(f"Error generating description for {self.image_path}: {e}")

    def get_datetime(self) -> Optional[str]:
        """Get the datetime for the image, preferring EXIF data if available."""
        exif_datetime = self._get_image_datetime(self.image_path)
        return exif_datetime if exif_datetime else self._get_file_datetime(self.image_path)

    @staticmethod
    def _get_image_datetime(image_path: str) -> Optional[str]:
        """Extract the datetime from the EXIF metadata of the image."""
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    if tag_name == "DateTimeOriginal":
                        return value
            return None
        except Exception as e:
            print(f"Error extracting EXIF data: {e}")
            return None

    @staticmethod
    def _get_file_datetime(image_path: str) -> Optional[str]:
        """Get the file's last modification date and time."""
        try:
            mod_time = os.path.getmtime(image_path)
            return datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error getting file datetime: {e}")