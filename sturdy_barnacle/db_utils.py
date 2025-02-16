import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml
from pgvector.sqlalchemy import Vector
from PIL import ExifTags, Image, TiffImagePlugin
from sqlalchemy import JSON, Column, Integer, String, Text, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker


def validate_table_names(table_dict: dict) -> dict:
    """Ensures all table names are safe from SQL injection attacks."""
    valid_pattern = re.compile(
        r"^[a-zA-Z0-9_]+$"
    )  # Allow only letters, numbers, and underscores

    for _, table_name in table_dict.items():
        if not valid_pattern.match(table_name):
            raise ValueError(f"Invalid table name detected: {table_name}")

    print("All table names are validated and safe.")
    return table_dict


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_URL = CONFIG["database"]["db_url"]
TABLES = validate_table_names(CONFIG["database"]["table_names"])

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
    ocr_text = Column(Text, nullable=True)
    search_vector = Column(Text, nullable=True)


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
        ocr_text: Optional[str] = None,
        search_vector: Optional[str] = None,
    ) -> None:
        """Saves or updates image metadata in the database."""
        session = self._get_session()
        try:

            exif_data = self.extract_exif_data(image_path)
            datetime = (
                exif_data.get("DateTimeOriginal", None) if exif_data else None
            )

            existing_image = (
                session.query(ImageMetadata)
                .filter_by(image_path=image_path)
                .first()
            )

            update_fields = {
                "description": description,
                "detected_objects": detected_objects,
                "exif_data": exif_data,
                "datetime": datetime,
                "embedding": embedding,
                "ocr_text": ocr_text,
                "search_vector": search_vector,
            }

            if existing_image:
                for field, value in update_fields.items():
                    if value is not None:
                        setattr(existing_image, field, value)
                print(f"Updated metadata for {image_path}")
            else:

                session.add(
                    ImageMetadata(
                        image_path=image_path,
                        **{
                            k: v
                            for k, v in update_fields.items()
                            if v is not None
                        },
                    )
                )

            session.commit()

        except Exception as e:
            session.rollback()
            print(f"Error saving/updating image metadata: {e}")
        finally:
            session.close()

    def query_images_by_keyword(self, keyword: str) -> List[ImageMetadata]:
        session = self._get_session()
        try:
            safe_keyword = re.sub(r"[^a-zA-Z0-9 ]", "", keyword)
            search_pattern = f"%{safe_keyword}%"
            return (
                session.query(ImageMetadata)
                .filter(
                    (ImageMetadata.description.ilike(text(":keyword")))
                    | (ImageMetadata.detected_objects.ilike(text(":keyword")))
                    | (
                        ImageMetadata.exif_data.cast(Text).ilike(
                            text(":keyword")
                        )
                    )
                )
                .params(keyword=search_pattern)
                .limit(
                    50
                )  # Limit query results to prevent excessive data leaks
                .all()
            )
        finally:
            session.close()

    def find_similar_images(
        self, image_embedding: List[float], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar images using embedding similarity search (pgvector)."""

        session = self._get_session()
        table_name = TABLES.get("image_metadata")
        try:
            embedding_str = f"[{', '.join(map(str, image_embedding))}]"
            query = text(
                f"""
            SELECT image_path, 1 - (embedding <=> :embedding) AS similarity
            FROM {table_name}
            ORDER BY similarity DESC
            LIMIT :top_n
        """
            )

            results = session.execute(
                query, {"embedding": embedding_str, "top_n": top_n}
            ).fetchall()

            return [
                {"image_path": row[0], "similarity": row[1]} for row in results
            ]

        finally:
            session.close()

    def create_album(self, album_name: str) -> int:
        session = self._get_session()
        try:
            table_name = TABLES.get("image_albums")
            query = text(
                f"""
            INSERT INTO {table_name} (album_name)
            VALUES (:name) RETURNING id
            """
            )

            result = session.execute(query, {"name": album_name})
            session.commit()
            return result.fetchone()[0]
        finally:
            session.close()

    def add_image_to_album(self, album_id: int, image_path: str):
        """Links an image to an album."""
        session = self._get_session()
        try:
            table_name = TABLES.get("image_album_mapping")
            query = text(
                f"""
                INSERT INTO {table_name} (album_id, image_path)
                VALUES (:album_id, :image_path)
                ON CONFLICT (image_path)
                DO UPDATE SET album_id = EXCLUDED.album_id;
            """
            )
            session.execute(
                query,
                {
                    "album_id": album_id,
                    "image_path": image_path,
                },
            )
            session.commit()
        finally:
            session.close()

    def get_album_images(self, album_id: int) -> List[str]:
        """Returns all images in an album."""
        session = self._get_session()
        try:
            table_name = TABLES.get("image_album_mapping")
            query = text(
                f"""SELECT image_path FROM {table_name}
                WHERE album_id = :album_id"""
            )
            results = session.execute(
                query,
                {
                    "album_id": album_id,
                },
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

    def delete_image_by_path(self, image_path: str) -> None:
        session = self._get_session()
        try:
            session.query(ImageMetadata).filter_by(
                image_path=image_path
            ).delete()
            print(f"Deleted metadata for {image_path}")
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error deleting image metadata: {e}")
        finally:
            session.close()
