import sqlite3
from typing import Dict, Optional

class DatabaseManager:
    """Handles SQLite database interactions for storing and retrieving image processing results."""
    
    def __init__(self, db_path: str = "image_data.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database and create necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT UNIQUE,
                    description TEXT,
                    detected_objects TEXT,
                    datetime TEXT
                )
            """)
            conn.commit()

    def save_results(self, image_path: str, description: str, detected_objects: Dict[str, int], datetime: Optional[str]):
        """Insert or update image processing results."""
        detected_objects_str = ", ".join([f"{obj} ({count})" for obj, count in detected_objects.items()])
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO images (image_path, description, detected_objects, datetime)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_path) DO UPDATE SET
                    description = excluded.description,
                    detected_objects = excluded.detected_objects,
                    datetime = excluded.datetime
            """, (image_path, description, detected_objects_str, datetime))
            conn.commit()

    def get_image_data(self, image_path: str):
        """Retrieve image processing results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images WHERE image_path = ?", (image_path,))
            result = cursor.fetchone()
            return result
        
    def is_image_processed(self, image_path: str) -> bool:
        """Check if image processing results are already stored."""
        return self.get_image_data(image_path) is not None