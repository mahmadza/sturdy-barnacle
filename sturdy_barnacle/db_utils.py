import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml
from pgvector.sqlalchemy import Vector
from PIL import ExifTags, Image, TiffImagePlugin
from sqlalchemy import Column, Integer, String, Text, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_URL = CONFIG["database"]["db_url"]
TABLES = CONFIG["database"]["table_names"]


def validate_table_names(table_dict: dict) -> dict:
    """Ensures all table names are safe from SQL injection attacks."""
    valid_pattern = re.compile(r"^[a-zA-Z0-9_]+$")
    for _, table_name in table_dict.items():
        if not valid_pattern.match(table_name):
            raise ValueError(f"Invalid table name detected: {table_name}")
    print("All table names validated.")
    return table_dict


TABLES = validate_table_names(TABLES)

# SQLAlchemy setup
engine = create_engine(DB_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class ImageMetadata(Base):
    __tablename__ = TABLES["image_metadata"]

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    detected_objects = Column(Text, nullable=True)
    datetime = Column(String, nullable=True)
    embedding = Column(Vector(512))
    ocr_text = Column(Text, nullable=True)


class DatabaseManager:
    """Manages all database interactions."""

    def __init__(self):
        self.db_name = self._get_database_name()
        self.engine = engine
        self.SessionLocal = SessionLocal
        self._initialize_db()

    def _get_database_name(self) -> str:
        return urlparse(DB_URL).path.lstrip("/")

    def _initialize_db(self) -> None:
        """Creates tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)

    def _get_session(self) -> Session:
        return self.SessionLocal()

    def is_image_processed(self, image_path: str) -> bool:
        """Check if image already exists in the database."""
        with self._get_session() as session:
            return (
                session.query(ImageMetadata)
                .filter_by(image_path=image_path)
                .first()
                is not None
            )

    def save_image_metadata(
        self,
        image_path: str,
        description: Optional[str] = None,
        detected_objects: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        ocr_text: Optional[str] = None,
    ) -> None:
        """Save or update image metadata."""
        with self._get_session() as session:
            exif_data = self.extract_exif_data(image_path)
            datetime = exif_data.get("DateTimeOriginal") if exif_data else None

            existing_image = (
                session.query(ImageMetadata)
                .filter_by(image_path=image_path)
                .first()
            )

            if existing_image:
                self._update_existing_image(
                    existing_image,
                    description,
                    detected_objects,
                    datetime,
                    embedding,
                    ocr_text,
                )
                print(f"Updated metadata for {image_path}")
            else:
                self._insert_new_image(
                    session,
                    image_path,
                    description,
                    detected_objects,
                    datetime,
                    embedding,
                    ocr_text,
                )

            session.commit()

    def _update_existing_image(
        self,
        image,
        description,
        detected_objects,
        datetime,
        embedding,
        ocr_text,
    ):
        update_fields = {
            "description": description,
            "detected_objects": detected_objects,
            "datetime": datetime,
            "embedding": embedding,
            "ocr_text": ocr_text,
        }
        for field, value in update_fields.items():
            if value is not None:
                setattr(image, field, value)

    def _insert_new_image(
        self,
        session,
        image_path,
        description,
        detected_objects,
        datetime,
        embedding,
        ocr_text,
    ):
        session.add(
            ImageMetadata(
                image_path=image_path,
                description=description,
                detected_objects=detected_objects,
                datetime=datetime,
                embedding=embedding,
                ocr_text=ocr_text,
            )
        )

    def extract_exif_data(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract EXIF metadata."""
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    return {
                        ExifTags.TAGS.get(tag, tag): self._convert_exif_value(
                            value
                        )
                        for tag, value in exif_data.items()
                    }
        except Exception as e:
            print(f"Error extracting EXIF from {image_path}: {e}")
        return None

    @staticmethod
    def _convert_exif_value(value: Any) -> Any:
        """Make EXIF data JSON serializable."""
        if isinstance(value, TiffImagePlugin.IFDRational):
            return float(value)
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if isinstance(value, tuple):
            return tuple(DatabaseManager._convert_exif_value(v) for v in value)
        if isinstance(value, dict):
            return {
                k: DatabaseManager._convert_exif_value(v)
                for k, v in value.items()
            }
        return value

    def query_images_by_keyword(self, keyword: str) -> List[ImageMetadata]:
        """Search images by keyword in description, objects, or OCR text."""
        safe_keyword = re.sub(r"[^a-zA-Z0-9 ]", "", keyword)
        search_pattern = f"%{safe_keyword}%"

        with self._get_session() as session:
            return (
                session.query(ImageMetadata)
                .filter(
                    ImageMetadata.description.ilike(text(":keyword"))
                    | ImageMetadata.detected_objects.ilike(text(":keyword"))
                    | ImageMetadata.ocr_text.ilike(text(":keyword"))
                )
                .params(keyword=search_pattern)
                .limit(50)
                .all()
            )

    def find_similar_images(
        self, embedding: List[float], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar images using pgvector cosine similarity."""
        embedding_str = f"[{', '.join(map(str, embedding))}]"

        query = text(
            f"""
            SELECT image_path, 1 - (embedding <=> :embedding) AS similarity
            FROM {TABLES['image_metadata']}
            ORDER BY similarity DESC
            LIMIT :top_n
        """
        )

        with self._get_session() as session:
            results = session.execute(
                query, {"embedding": embedding_str, "top_n": top_n}
            ).fetchall()

        return [
            {"image_path": row[0], "similarity": row[1]} for row in results
        ]

    def query_all_images(self) -> List[ImageMetadata]:
        """Fetch all images."""
        with self._get_session() as session:
            return session.query(ImageMetadata).all()

    def get_image_by_path(self, image_path: str) -> Optional[ImageMetadata]:
        """Fetch image metadata by path."""
        with self._get_session() as session:
            return (
                session.query(ImageMetadata)
                .filter_by(image_path=image_path)
                .first()
            )

    def delete_image_by_path(self, image_path: str) -> None:
        """Delete image by path."""
        with self._get_session() as session:
            session.query(ImageMetadata).filter_by(
                image_path=image_path
            ).delete()
            session.commit()
            print(f"Deleted metadata for {image_path}")
