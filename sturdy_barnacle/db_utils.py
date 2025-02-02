from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml
from pgvector.sqlalchemy import Vector
from PIL import ExifTags, Image, TiffImagePlugin
from sqlalchemy import JSON, Column, Integer, String, Text, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_URL = CONFIG["database"]["db_url"]
TABLES = CONFIG["database"]["table_names"]

engine = create_engine(DB_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ImageMetadata(Base):

    __tablename__ = TABLES["image_metadata"]

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    detected_objects = Column(Text, nullable=True)
    datetime = Column(String, nullable=True)
    exif_data = Column(JSON, nullable=True)
    embedding = Column(Vector(512))


class DatabaseManager:
    """Handles all database operations."""

    def __init__(self):
        self.db_url = DB_URL
        self.db_name = self.get_database_name()
        self.engine = engine
        self.SessionLocal = SessionLocal
        self._initialize_db()

    def get_database_name(self) -> str:
        """Extracts the database name from the URL."""
        return urlparse(self.db_url).path.lstrip("/")

    def _initialize_db(self) -> None:
        """Creates database tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    def _get_session(self) -> Session:
        """Returns a new database session."""
        return self.SessionLocal()

    def is_image_processed(self, image_path: str) -> bool:
        """Checks if an image is already in the database."""
        session = self._get_session()
        try:
            result = (
                session.query(ImageMetadata)
                .filter_by(image_path=image_path)
                .first()
            )
            return result is not None
        finally:
            session.close()

    @staticmethod
    def _convert_exif_value(value: Any) -> Any:
        """Converts EXIF values to JSON-serializable format."""
        if isinstance(value, TiffImagePlugin.IFDRational):
            return float(value)
        elif isinstance(value, bytes):
            return value.decode(errors="replace")
        elif isinstance(value, tuple):
            return tuple(DatabaseManager._convert_exif_value(v) for v in value)
        elif isinstance(value, dict):
            return {
                k: DatabaseManager._convert_exif_value(v)
                for k, v in value.items()
            }
        return value

    def extract_exif_data(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extracts EXIF metadata from an image."""
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data:
                return {
                    ExifTags.TAGS.get(tag, tag): self._convert_exif_value(
                        value
                    )
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
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Saves or updates image metadata in the database."""
        session = self._get_session()
        try:
            if self.is_image_processed(image_path):
                print(f"Skipping {image_path} (already processed)")
                return

            exif_data = self.extract_exif_data(image_path)
            datetime = (
                exif_data.get("DateTimeOriginal", None) if exif_data else None
            )

            existing = (
                session.query(ImageMetadata)
                .filter_by(image_path=image_path)
                .first()
            )
            if existing:
                existing.description = description
                existing.detected_objects = detected_objects
                existing.exif_data = exif_data
                if embedding:
                    existing.embedding = embedding
            else:
                session.add(
                    ImageMetadata(
                        image_path=image_path,
                        description=description,
                        detected_objects=detected_objects,
                        datetime=datetime,
                        exif_data=exif_data,
                        embedding=embedding,
                    )
                )

            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving image metadata: {e}")
        finally:
            session.close()

    def query_images_by_keyword(self, keyword: str) -> List[ImageMetadata]:
        session = self._get_session()
        try:
            return (
                session.query(ImageMetadata)
                .filter(
                    (ImageMetadata.description.ilike(f"%{keyword}%"))
                    | (ImageMetadata.detected_objects.ilike(f"%{keyword}%"))
                    | (
                        ImageMetadata.exif_data.cast(Text).ilike(
                            f"%{keyword}%"
                        )
                    )
                )
                .all()
            )
        finally:
            session.close()

    def find_similar_images(
        self, image_embedding: List[float], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar images using embedding similarity search (pgvector)."""
        session = self._get_session()
        try:
            embedding_str = (
                f"'[{', '.join(map(str, image_embedding))}]'::vector"
            )
            query = text(
                f"""
                SELECT image_path, 1 - (embedding <=> {embedding_str})
                AS similarity
                FROM {TABLES["image_metadata"]}
                ORDER BY similarity DESC
                LIMIT :top_n
            """
            )
            results = session.execute(query, {"top_n": top_n}).fetchall()
            return [
                {"image_path": row[0], "similarity": row[1]} for row in results
            ]
        finally:
            session.close()

    def create_album(self, album_name: str) -> int:
        session = self._get_session()
        try:
            result = session.execute(
                text(
                    f"""INSERT INTO {TABLES['image_albums']} (album_name)
                    VALUES (:name) RETURNING id"""
                ),
                {"name": album_name},
            )
            session.commit()
            return result.fetchone()[0]
        finally:
            session.close()

    def add_image_to_album(self, album_id: int, image_path: str):
        """Links an image to an album."""
        session = self._get_session()
        try:
            session.execute(
                text(
                    f"""
                    INSERT INTO {TABLES['image_album_mapping']}
                    (album_id, image_path)
                    VALUES (:album_id, :image_path)
                    ON CONFLICT (image_path)
                    DO UPDATE SET album_id = EXCLUDED.album_id;
                """
                ),
                {"album_id": album_id, "image_path": image_path},
            )
            session.commit()
        finally:
            session.close()

    def get_album_images(self, album_id: int) -> List[str]:
        """Returns all images in an album."""
        session = self._get_session()
        try:
            results = session.execute(
                text(
                    f"""SELECT image_path FROM {TABLES['image_album_mapping']}
                    WHERE album_id = :album_id"""
                ),
                {"album_id": album_id},
            )
            return [row[0] for row in results]
        finally:
            session.close()

    def query_all_images(self) -> List[ImageMetadata]:
        """Returns all stored images."""
        session = self._get_session()
        try:
            return session.query(ImageMetadata).all()
        finally:
            session.close()

    def get_image_by_path(self, image_path: str) -> Optional[ImageMetadata]:
        """Returns image metadata by path."""
        session = self._get_session()
        try:
            return (
                session.query(ImageMetadata)
                .filter_by(image_path=image_path)
                .first()
            )
        finally:
            session.close()
