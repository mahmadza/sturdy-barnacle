from sqlalchemy import create_engine, Column, Integer, String, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pgvector.sqlalchemy import Vector
from PIL import Image, ExifTags
from typing import Optional, List, Dict, Any
import json

DB_URL = "postgresql://myuser:mypassword@localhost:5432/images_db"

engine = create_engine(DB_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ImageMetadata(Base):
    """ORM Model for storing image metadata, EXIF data (JSON), and embeddings."""
    __tablename__ = "image_metadata"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    detected_objects = Column(Text, nullable=True)
    datetime = Column(String, nullable=True)
    exif_data = Column(JSON, nullable=True)
    embedding = Column(Vector(512))

class DatabaseManager:

    def __init__(self, db_url: str = DB_URL):
        self.engine = create_engine(db_url, pool_size=10, max_overflow=20)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._initialize_db()

    def _initialize_db(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    def _get_session(self) -> Session:
        return self.SessionLocal()

    def is_image_processed(self, image_path: str) -> bool:
        """Check if an image has already been processed and exists in the database."""
        session = self._get_session()
        try:
            return session.query(ImageMetadata).filter_by(image_path=image_path).first() is not None
        finally:
            session.close()

    def _convert_exif_value(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode(errors="ignore")  # Convert bytes to string
        elif isinstance(value, tuple):
            return tuple(map(self._convert_exif_value, value))  # Recursively convert tuples
        elif hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value.numerator) / float(value.denominator)  # Convert IFDRational to float
        return value  # Return as is if already serializable

    def extract_exif_data(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract EXIF metadata from an image and ensure it is JSON serializable."""
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
        """Save or update image metadata, including EXIF data and embeddings."""
        session = self._get_session()
        
        try:
            
            if self.is_image_processed(image_path):
                print(f"Skipping {image_path} (already processed)")
                return
            
            exif_data = self.extract_exif_data(image_path)
            datetime = exif_data.get("DateTimeOriginal", None)
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
            results = session.execute(f"""
                SELECT image_path, 1 - (embedding <=> '{image_embedding}') AS similarity 
                FROM image_metadata 
                ORDER BY similarity DESC 
                LIMIT {top_n}
            """).fetchall()
            return [{"image_path": row[0], "similarity": row[1]} for row in results]
        finally:
            session.close()