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
