import json
from collections import Counter
from pathlib import Path
from typing import List

import cv2
import torch
import yaml
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.model_zoo import model_zoo
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPModel,
    CLIPProcessor,
)

from sturdy_barnacle.db_utils import DatabaseManager

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

MODEL_NAMES = config["models"]


class ImageProcessor:
    """Processes images using BLIP-2, Detectron2, and CLIP models."""

    _resources_initialized = False

    def __init__(self, image_path: str, db: DatabaseManager) -> None:
        self.image_path: str = image_path
        self.db: DatabaseManager = db
        self.device = config["image_processing"]["default_device"]

        self.skip_processing = self.db.is_image_processed(self.image_path)

        if not ImageProcessor._resources_initialized:
            ImageProcessor._initialize_shared_resources()

    @classmethod
    def _initialize_shared_resources(cls) -> None:
        """Loads models once per execution for efficiency."""
        print("Initializing shared resources...")

        cls._initialize_blip()
        cls._initialize_detectron()
        cls._initialize_clip()

        cls._resources_initialized = True
        print("All models loaded successfully!")

    @classmethod
    def _initialize_blip(cls) -> None:
        """Initializes BLIP-2 model for image captioning."""
        try:
            model_name = MODEL_NAMES["blip"]
            cls._blip_processor = Blip2Processor.from_pretrained(model_name)
            cls._blip_model = Blip2ForConditionalGeneration.from_pretrained(
                model_name
            ).to(config["image_processing"]["default_device"])
        except Exception as e:
            raise RuntimeError(f"Error initializing BLIP-2 model: {e}")

    @classmethod
    def _initialize_detectron(cls) -> None:
        """Initializes Detectron2 for object detection."""
        try:
            model_name = MODEL_NAMES["detectron"]
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(model_name))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = config["image_processing"]["default_device"]

            cls._detectron_cfg = cfg
            cls._detectron_predictor = DefaultPredictor(cfg)
            cls._metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            cls._class_names: List[str] = cls._metadata.get(
                "thing_classes", []
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing Detectron2: {e}")

    @classmethod
    def _initialize_clip(cls) -> None:
        """Initializes CLIP model for embeddings."""
        try:
            model_name = MODEL_NAMES["clip"]
            cls._clip_processor = CLIPProcessor.from_pretrained(model_name)
            cls._clip_model = CLIPModel.from_pretrained(model_name).to(
                config["image_processing"]["default_device"]
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing CLIP model: {e}")

    def detect_objects(self) -> Counter[str]:
        """Detects objects in an image using Detectron2."""
        if not hasattr(ImageProcessor, "_detectron_predictor"):
            raise RuntimeError("Detectron2 model is not initialized.")

        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not read image at {self.image_path}")

        outputs = ImageProcessor._detectron_predictor(image)
        instances = outputs["instances"].to(
            config["image_processing"]["default_device"]
        )
        pred_classes = instances.pred_classes.numpy()
        detected_items: List[str] = [
            ImageProcessor._class_names[c] for c in pred_classes
        ]
        return Counter(detected_items)

    def describe_image(self) -> str:
        """Generates an image caption using BLIP-2."""
        if not hasattr(ImageProcessor, "_blip_model"):
            raise RuntimeError("BLIP models are not initialized.")

        try:
            image = Image.open(self.image_path).convert("RGB")
            inputs = ImageProcessor._blip_processor(
                image, return_tensors="pt"
            ).to(self.device)
            outputs = ImageProcessor._blip_model.generate(**inputs)
            return ImageProcessor._blip_processor.decode(
                outputs[0], skip_special_tokens=True
            )
        except Exception as e:
            raise ValueError(
                f"Error generating description for {self.image_path}: {e}"
            )

    def compute_embedding(self) -> List[float]:
        """Computes an image embedding using CLIP."""
        if not hasattr(ImageProcessor, "_clip_model"):
            raise RuntimeError("CLIP model is not initialized.")

        try:
            image = Image.open(self.image_path).convert("RGB")
            inputs = ImageProcessor._clip_processor(
                images=image, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                embedding = (
                    ImageProcessor._clip_model.get_image_features(**inputs)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .tolist()
                )
            return embedding
        except Exception as e:
            raise ValueError(
                f"Error computing embedding for {self.image_path}: {e}"
            )

    def process_and_store(self) -> None:
        """Processes an image and stores metadata in the database."""
        if self.skip_processing:
            print(f"Skipping {self.image_path} (already processed)")
            return

        description = self.describe_image()
        detected_objects = self.detect_objects()
        embedding = self.compute_embedding()
        detected_objects_str = json.dumps(detected_objects)

        self.db.save_image_metadata(
            image_path=self.image_path,
            description=description,
            detected_objects=detected_objects_str,
            embedding=embedding,
        )
