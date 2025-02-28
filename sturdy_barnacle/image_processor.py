import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
import yaml
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.model_zoo import model_zoo
from paddleocr import PaddleOCR
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPModel,
    CLIPProcessor,
)

from sturdy_barnacle.db_utils import DatabaseManager

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

MODEL_NAMES = config["models"]


class ImageProcessor:
    """Processes images using BLIP-2, Detectron2, CLIP, and PaddleOCR."""

    # Shared class-level resources (initialized once)
    _resources_initialized = False
    _blip_processor: Optional[Blip2Processor] = None
    _blip_model: Optional[Blip2ForConditionalGeneration] = None
    _detectron_predictor: Optional[DefaultPredictor] = None
    _metadata: Optional[Dict] = None
    _class_names: List[str] = []
    _clip_processor: Optional[CLIPProcessor] = None
    _clip_model: Optional[CLIPModel] = None
    _fastocr: Optional[PaddleOCR] = None

    def __init__(self, image_path: str, db: DatabaseManager) -> None:
        self.image_path: str = image_path
        self.db: DatabaseManager = db

        if not ImageProcessor._resources_initialized:
            self._initialize_shared_resources()

    @classmethod
    def _initialize_shared_resources(cls) -> None:
        """Initialize shared resources (models, configs) once per process."""
        print("Initializing shared resources...")
        cls._set_device()
        cls._initialize_blip()
        cls._initialize_detectron()
        cls._initialize_clip()
        cls._initialize_fastocr()
        cls._resources_initialized = True
        print("All models loaded successfully!")

    @classmethod
    def _set_device(cls) -> None:
        """Set device preferences for each model."""
        cls.device = config["device"]["default_device"]

        if cls.device == "mps":
            print("MPS selected. Detectron2 will be forced to use CPU.")
            cls.detectron_device = "cpu"
        else:
            cls.detectron_device = cls.device

    @classmethod
    def _initialize_blip(cls) -> None:
        """Initialize BLIP-2 for captioning."""
        try:
            model_name = MODEL_NAMES["blip"]
            cls._blip_processor = Blip2Processor.from_pretrained(model_name)
            cls._blip_model = Blip2ForConditionalGeneration.from_pretrained(
                model_name
            ).to(cls.device)
        except Exception as e:
            raise RuntimeError(f"Error initializing BLIP-2: {e}")

    @classmethod
    def _initialize_detectron(cls) -> None:
        """Initialize Detectron2 for object detection."""
        try:
            model_name = MODEL_NAMES["detectron"]
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(model_name))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = cls.detectron_device

            cls._detectron_predictor = DefaultPredictor(cfg)
            cls._metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            cls._class_names = cls._metadata.get("thing_classes", [])
        except Exception as e:
            raise RuntimeError(f"Error initializing Detectron2: {e}")

    @classmethod
    def _initialize_clip(cls) -> None:
        """Initialize CLIP for embeddings."""
        try:
            model_name = MODEL_NAMES["clip"]
            cls._clip_processor = CLIPProcessor.from_pretrained(model_name)
            cls._clip_model = CLIPModel.from_pretrained(model_name).to(
                cls.device
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing CLIP: {e}")

    @classmethod
    def _initialize_fastocr(cls) -> None:
        """Initialize PaddleOCR for text extraction."""
        try:
            if torch.cuda.is_available():
                use_gpu = True
            elif torch.backends.mps.is_available():
                use_gpu = (
                    False  # PaddleOCR does not support MPS, fallback to CPU
                )
            else:
                use_gpu = False

            logging.getLogger("ppocr").setLevel(
                logging.WARNING
            )  # Suppress PaddleOCR logs

            cls._fastocr = PaddleOCR(use_angle_cls=True, use_gpu=use_gpu)
            print(f"PaddleOCR initialized using {'GPU' if use_gpu else 'CPU'}")
        except Exception as e:
            raise RuntimeError(f"Error initializing PaddleOCR: {e}")

    def detect_objects(self) -> Counter[str]:
        """Detect objects using Detectron2."""
        image = self._load_image()

        outputs = ImageProcessor._detectron_predictor(image)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.cpu().numpy()

        detected_items = [ImageProcessor._class_names[c] for c in pred_classes]
        return Counter(detected_items)

    def describe_image(self) -> str:
        """Generate image caption using BLIP-2."""
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
        """Compute image embedding using CLIP."""
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

    def extract_text(self) -> str:
        """Extract text using PaddleOCR."""
        try:
            image = self._load_image(rgb=True)
            results = ImageProcessor._fastocr.ocr(image, cls=True)

            if not results or results[0] is None:
                return ""

            ocr_text = " ".join([res[1][0] for res in results[0] if res[1]])
            return ocr_text.strip()
        except Exception as e:
            print(f"OCR failed for {self.image_path}: {e}")
            return ""

    def process_and_store(self) -> None:
        """Process the image and store metadata."""
        description = self.describe_image()
        detected_objects = self.detect_objects()
        embedding = self.compute_embedding()
        ocr_text = self.extract_text()
        detected_objects_str = json.dumps(detected_objects)

        self.db.save_image_metadata(
            image_path=self.image_path,
            description=description,
            detected_objects=detected_objects_str,
            embedding=embedding,
            ocr_text=ocr_text,
        )

    def _load_image(self, rgb=False) -> Image:
        """Load image using OpenCV and handle RGB conversion if needed."""
        image = cv2.imread(self.image_path)
        self._validate_image_size(image)
        if image is None:
            raise ValueError(f"Could not read image at {self.image_path}")

        if rgb:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _validate_image_size(self, image) -> None:
        """Validate image size is at least 32x32."""
        h, w = image.shape[:2]
        if h < 32 or w < 32:
            raise ValueError(
                f"Image too small: {h}x{w}. Minimum size is 32x32."
            )
