from sturdy_barnacle.db_utils import DatabaseManager
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import torch
import json
from collections import Counter
from typing import List

class ImageProcessor:

    def __init__(self, image_path: str, db: DatabaseManager) -> None:
        self.image_path: str = image_path
        self.db: DatabaseManager = db
        
        if self.db.is_image_processed(self.image_path):
            print(f"Skipping {self.image_path} (already processed)")
            self.skip_processing = True
        else:
            self.skip_processing = False
            self._initialize_shared_resources()


    @classmethod
    def _initialize_shared_resources(cls) -> None:

        if not hasattr(cls, "_blip_model"):
            cls._initialize_blip()
        if not hasattr(cls, "_detectron_predictor"):
            cls._initialize_detectron()
        if not hasattr(cls, "_clip_model"):
            cls._initialize_clip()


    @classmethod
    def _initialize_blip(cls) -> None:

        try:
            cls._blip_processor: Blip2Processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            cls._blip_model: Blip2ForConditionalGeneration = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        except Exception as e:
            raise RuntimeError(f"Error initializing BLIP-2 model: {e}")


    @classmethod
    def _initialize_detectron(cls) -> None:

        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            cls._detectron_cfg = cfg
            cls._detectron_predictor: DefaultPredictor = DefaultPredictor(cfg)
            cls._metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            cls._class_names: List[str] = cls._metadata.get("thing_classes", [])
        except Exception as e:
            raise RuntimeError(f"Error initializing Detectron2: {e}")


    @classmethod
    def _initialize_clip(cls) -> None:

        try:
            cls._clip_model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            cls._clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            raise RuntimeError(f"Error initializing CLIP model: {e}")


    def detect_objects(self) -> Counter[str]:

        if not hasattr(self, "_detectron_predictor"):
            raise RuntimeError("Detectron2 model is not initialized.")

        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not read image at {self.image_path}")

        outputs = self._detectron_predictor(image)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.numpy()
        detected_items: List[str] = [self._class_names[c] for c in pred_classes]
        return Counter(detected_items)


    def describe_image(self) -> str:

        if not hasattr(self, "_blip_model"):
            raise RuntimeError("BLIP models are not initialized.")

        try:
            image: Image.Image = Image.open(self.image_path).convert("RGB")
            inputs = self._blip_processor(image, return_tensors="pt")
            outputs = self._blip_model.generate(**inputs)
            return self._blip_processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise ValueError(f"Error generating description for {self.image_path}: {e}")


    def compute_embedding(self) -> List[float]:

        if not hasattr(self, "_clip_model"):
            raise RuntimeError("CLIP model is not initialized.")

        try:
            image: Image.Image = Image.open(self.image_path).convert("RGB")
            inputs = self._clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding: List[float] = self._clip_model.get_image_features(**inputs).squeeze(0).cpu().numpy().tolist()
            return embedding
        except Exception as e:
            raise ValueError(f"Error computing embedding for {self.image_path}: {e}")


    def process_and_store(self) -> None:

        if self.skip_processing:
            return        
        
        description: str = self.describe_image()
        detected_objects: Counter[str] = self.detect_objects()
        embedding: List[float] = self.compute_embedding()
        detected_objects_str: str = json.dumps(detected_objects)

        self.db.save_image_metadata(
            image_path=self.image_path,
            description=description,
            detected_objects=detected_objects_str,
            embedding=embedding
        )
