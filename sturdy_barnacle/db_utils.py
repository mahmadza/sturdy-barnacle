from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from PIL import Image, ExifTags
import psycopg2

# PostgreSQL Database Config (Modify as needed)
DB_URL = "postgresql://user:pass@localhost:5432/images_db"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    detected_objects = Column(Text, nullable=True)
    datetime = Column(String, nullable=True)

    # EXIF Metadata Fields
    camera_make = Column(String, nullable=True)
    camera_model = Column(String, nullable=True)
    lens_model = Column(String, nullable=True)
    exposure_time = Column(String, nullable=True)
    f_number = Column(Float, nullable=True)
    iso = Column(Integer, nullable=True)
    focal_length = Column(Float, nullable=True)
    gps_latitude = Column(Float, nullable=True)
    gps_longitude = Column(Float, nullable=True)
    gps_altitude = Column(Float, nullable=True)

def initialize_db():
    Base.metadata.create_all(bind=engine)

def update_exif_data(image_path):
    """Extract EXIF data and update PostgreSQL for an existing image."""
    session = SessionLocal()
    
    # Check if the image exists in the database
    image_record = session.query(ImageRecord).filter_by(image_path=image_path).first()
    if not image_record:
        print(f"Image not found in DB: {image_path}")
        session.close()
        return

    # Extract EXIF metadata
    exif_data = extract_exif_data(image_path)

    # Update only EXIF fields, keep existing descriptions/detections
    image_record.camera_make = exif_data.get("camera_make")
    image_record.camera_model = exif_data.get("camera_model")
    image_record.lens_model = exif_data.get("lens_model")
    image_record.exposure_time = exif_data.get("exposure_time")
    image_record.f_number = exif_data.get("f_number")
    image_record.iso = exif_data.get("iso")
    image_record.focal_length = exif_data.get("focal_length")
    image_record.gps_latitude = exif_data.get("gps_latitude")
    image_record.gps_longitude = exif_data.get("gps_longitude")
    image_record.gps_altitude = exif_data.get("gps_altitude")

    session.commit()
    session.close()
    print(f"Updated EXIF for: {image_path}")
