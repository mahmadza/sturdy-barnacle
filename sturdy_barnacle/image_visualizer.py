import json

import matplotlib.pyplot as plt
from PIL import Image

from sturdy_barnacle.db_utils import DatabaseManager


class ImageVisualizer:
    """Class to visualize images and their metadata from the database."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def display_image(self, image_path: str) -> None:
        try:
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Error displaying image {image_path}: {e}")

    def visualize_image_metadata(self, image_path: str) -> None:

        img_metadata = self.db.get_image_by_path(image_path)
        if not img_metadata:
            print(f"No metadata found for image: {image_path}")
            return

        print(f"Image Path: {img_metadata.image_path}")

        self.display_image(img_metadata.image_path)

        detected_objects = (
            json.loads(img_metadata.detected_objects)
            if img_metadata.detected_objects
            else {}
        )
        print(f"Detected Objects: {detected_objects}")
        print(f"Caption: {img_metadata.description or 'No caption available'}")
        print(f"OCR Text: {img_metadata.ocr_text or 'No text detected'}")
        print(f"Date Taken: {img_metadata.datetime or 'Unknown'}")
        print("-" * 50)
