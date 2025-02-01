from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pgvector.sqlalchemy import Vector
from PIL import Image, ExifTags, TiffImagePlugin
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from pathlib import Path
import json
from sklearn.cluster import KMeans
import numpy as np

CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

DB_URL = CONFIG["database"]["db_url"]
TABLE_IMAGE_METADATA = CONFIG["tables"]["image_metadata"]


engine = create_engine(DB_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ImageMetadata(Base):
    """ORM Model for storing image metadata, EXIF data (JSON), and embeddings."""
    __tablename__ = TABLE_IMAGE_METADATA

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    detected_objects = Column(Text, nullable=True)
    datetime = Column(String, nullable=True)
    exif_data = Column(JSON, nullable=True)
    embedding = Column(Vector(512))

class DatabaseManager:

    def __init__(self):
        """Initialize the DatabaseManager and extract database name."""
        self.db_url = DB_URL
        self.db_name = self.get_database_name()

        self.engine = create_engine(self.db_url, pool_size=10, max_overflow=20)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_database_name(self) -> str:
        parsed_url = urlparse(self.db_url)
        return parsed_url.path.lstrip("/") 

    def _initialize_db(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    def _get_session(self) -> Session:
        return self.SessionLocal()

    def is_image_processed(self, image_path: str) -> bool:
        session = self._get_session()
        try:
            return session.query(ImageMetadata).filter_by(image_path=image_path).first() is not None
        finally:
            session.close()

    @staticmethod
    def _convert_exif_value(value: Any) -> Any:
        if isinstance(value, TiffImagePlugin.IFDRational):
            return float(value)  # Convert to float
        elif isinstance(value, bytes):
            return value.decode(errors="replace")  # Convert bytes to string
        elif isinstance(value, tuple):
            return tuple(DatabaseManager._convert_exif_value(v) for v in value)  # Convert tuple recursively
        elif isinstance(value, dict):
            return {k: DatabaseManager._convert_exif_value(v) for k, v in value.items()}  # Convert dict recursively
        return value  # Return as-is if already serializable

    def extract_exif_data(self, image_path: str) -> Optional[Dict[str, Any]]:
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data:
                return {
                    ExifTags.TAGS.get(tag, tag): self._convert_exif_value(value)
                    for tag, value in exif_data.items()
                }
        except Exception as e:
            print(f"Error extracting EXIF data from {image_path}: {e}")
        return None

    def save_image_metadata(
        self,
        image_path: str,
        description: Optional[str] = None,
        detected_objects: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        
        session = self._get_session()
        
        try:
            
            if self.is_image_processed(image_path):
                print(f"Skipping {image_path} (already processed)")
                return
            
            exif_data = self.extract_exif_data(image_path)
            datetime = exif_data.get("DateTimeOriginal", None) if exif_data else None
            existing = session.query(ImageMetadata).filter_by(image_path=image_path).first()

            if existing:
                existing.description = description
                existing.detected_objects = detected_objects
                existing.exif_data = exif_data
                if embedding is not None:
                    existing.embedding = embedding
            else:
                new_entry = ImageMetadata(
                    image_path=image_path,
                    description=description,
                    detected_objects=detected_objects,
                    datetime=datetime,
                    exif_data=exif_data,
                    embedding=embedding
                )
                session.add(new_entry)

            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving image metadata: {e}")
        finally:
            session.close()

    def query_images_by_keyword(self, keyword: str) -> List[ImageMetadata]:
        """Find images that match a keyword in description, detected objects, or EXIF data."""
        session = self._get_session()
        try:
            return session.query(ImageMetadata).filter(
                (ImageMetadata.description.ilike(f"%{keyword}%")) |
                (ImageMetadata.detected_objects.ilike(f"%{keyword}%")) |
                (ImageMetadata.exif_data.cast(Text).ilike(f"%{keyword}%"))
            ).all()
        finally:
            session.close()

    def find_similar_images(self, image_embedding: List[float], top_n: int = 5) -> List[Dict[str, Any]]:
        """Find similar images using embedding similarity search (pgvector)."""
        session = self._get_session()
        try:
            # Convert Python list to PostgreSQL-compatible vector format
            embedding_str = f"'[{', '.join(map(str, image_embedding))}]'::vector"

            query = text(f"""
                SELECT image_path, 1 - (embedding <=> {embedding_str}) AS similarity 
                FROM image_metadata 
                ORDER BY similarity DESC 
                LIMIT :top_n
            """)

            results = session.execute(query, {"top_n": top_n}).fetchall()
            return [{"image_path": row[0], "similarity": row[1]} for row in results]
        finally:
            session.close()

    def get_image_by_path(self, image_path: str) -> Optional[ImageMetadata]:
        session = self._get_session()
        try:
            return session.query(ImageMetadata).filter_by(image_path=image_path).first()
        finally:
            session.close()


    def create_album(self, album_name: str) -> int:
        """Creates a new album and returns the album ID."""
        session = self._get_session()
        try:
            result = session.execute(
                text("INSERT INTO image_albums (album_name) VALUES (:name) RETURNING id"),
                {"name": album_name}
            )
            session.commit()
            return result.fetchone()[0]
        finally:
            session.close()


    def add_image_to_album(self, album_id: int, image_path: str) -> None:
        """Adds an image to a specified album."""
        session = self._get_session()
        try:
            session.execute(
                text("INSERT INTO image_album_mapping (album_id, image_path) VALUES (:album_id, :image_path) ON CONFLICT DO NOTHING"),
                {"album_id": album_id, "image_path": image_path}
            )
            session.commit()
        finally:
            session.close()

    def get_album_images(self, album_id: int) -> List[str]:
        """Returns a list of image paths in a specified album."""
        session = self._get_session()
        try:
            results = session.execute(
                text("SELECT image_path FROM image_album_mapping WHERE album_id = :album_id"),
                {"album_id": album_id}
            )
            return [row[0] for row in results]
        finally:
            session.close()
    
    def query_all_images(self) -> List[ImageMetadata]:
        """Returns all images in the database."""
        session = self._get_session()
        try:
            return session.query(ImageMetadata).all()
        finally:
            session.close()


    def get_images_by_album(self, album_id: int) -> List[str]:
        """Returns all images in a specified album."""
        session = self._get_session()
        try:
            results = session.execute(
                text("SELECT image_path FROM image_album_mapping WHERE album_id = :album_id"),
                {"album_id": album_id}
            )
            return [row[0] for row in results]
        finally:
            session.close()