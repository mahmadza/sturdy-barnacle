# 📂 Sturdy Barnacle

**A Modern Image Processing & Album Management System**

Sturdy Barnacle is an **end-to-end image processing pipeline** that extracts metadata, captions, objects, embeddings, and text from images, stores them in **PostgreSQL with `pgvector`**, and clusters images into **intelligent albums** using **t-SNE & HDBSCAN**.
It supports **Detectron2, BLIP-2, CLIP, PaddleOCR**, and **PostgreSQL** for metadata storage, with robust similarity search and album summarization capabilities.

---

## 📑 Features

### ✅ Image Metadata Extraction
- **Captions** with BLIP-2 (image-to-text)
- **Object Detection** with Detectron2
- **Embeddings** with CLIP (for similarity search)
- **OCR Text Extraction** with PaddleOCR
- **EXIF Data Extraction** (camera data, GPS, timestamps)

### ✅ Metadata Storage & Search
- Stored in **PostgreSQL with pgvector**
- **Similarity Search** (find similar images using embeddings)
- **Keyword Search** (search captions, OCR, or object tags)

### ✅ Album Management
- Cluster images into albums using **t-SNE** + **HDBSCAN**
- Automatically name albums based on contents
- Generate **tags and summaries** from detected objects, captions, and EXIF data

---

## 📂 Folder Structure

```text
sturdy_barnacle/
├── __init__.py                    # Package initializer
├── config.yaml                     # Configuration file
├── db_utils.py                     # Database operations (PostgreSQL, pgvector, EXIF extraction)
├── image_processor.py              # Main image processing pipeline (caption, objects, OCR, embeddings)
├── image_visualizer.py             # Visualizer to display images + metadata
│
├── albums/                         # Album management components
│   ├── __init__.py
│   ├── album_manager.py            # t-SNE + HDBSCAN image clustering into albums
│   ├── auto_album_manager.py       # Auto-naming, tagging, summarization for albums
```

## 🧰 Technologies

| Component            | Technology                        |
|---------------------|----------------------------------|
| Database              | PostgreSQL + pgvector             |
| Image Captioning      | BLIP-2 (Hugging Face Transformers) |
| Object Detection      | Detectron2                        |
| Image Embedding       | CLIP (OpenAI)                     |
| OCR                   | PaddleOCR                         |
| Clustering            | t-SNE + HDBSCAN                   |
| Metadata Management   | SQLAlchemy                        |

---

## ⚙️ Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/sturdy_barnacle.git
cd sturdy_barnacle
```

### 2️⃣ Install Dependencies
Create a virtual environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Configure PostgreSQL
Ensure your PostgreSQL database has the pgvector extension installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4️⃣ Configure config.yaml
Edit the configuration file located at sturdy_barnacle/config.yaml to match your environment, for example:

```yaml
database:
  db_url: "postgresql://myuser:mypassword@localhost:5432/images_db"
  table_names:
    image_metadata: "image_metadata"
    image_albums: "image_albums"
    image_album_mapping: "image_album_mapping"

models:
  blip: "Salesforce/blip2-opt-2.7b"
  detectron: "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
  clip: "openai/clip-vit-base-patch32"

device:
  default_device: "mps"  # Options: mps, cuda, cpu
```

## 🚀 Usage
### 1️⃣ Initialize Database
```python
from sturdy_barnacle.db_utils import DatabaseManager

db = DatabaseManager()
```

### 2️⃣ Process an Image
```python
from sturdy_barnacle.image_processor import ImageProcessor

processor = ImageProcessor("path/to/image.jpg", db)
processor.process_and_store()
```

### 3️⃣ Cluster Images into Albums
```python
from sturdy_barnacle.albums.album_manager import AlbumManager

album_manager = AlbumManager(db)
album_manager.group_images_into_albums()
```

### 4️⃣ Auto-Name and Summarize Albums
```python
from sturdy_barnacle.albums.auto_album_manager import AutoAlbumManager

auto_album_manager = AutoAlbumManager(db)
auto_album_manager.auto_name_and_tag_albums()
```

### 5️⃣ Visualize Image Metadata
```python
from sturdy_barnacle.image_processor import ImageProcessor

processor = ImageProcessor("path/to/image.jpg", db)
processor.visualize_image_metadata()
```

### 📊 Data Flow
```text
                    Image File
                          |
                 ImageProcessor
        (Detectron2, BLIP-2, CLIP, PaddleOCR)
                          |
                        DB (PostgreSQL + pgvector)
                          |
           +----------------------------------+
           |                                  |
   AlbumManager                    ImageProcessor (Visualizer)
  (t-SNE + HDBSCAN)             (Display + Metadata View)
           |
   AutoAlbumManager
   (Album Naming + Tagging)
```

## 📄 Example Metadata Stored

| Field              | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `image_path`      | Full file path to the image.                                                |
| `description`     | AI-generated caption describing the image (from BLIP-2).                    |
| `detected_objects`| List of objects detected in the image (from Detectron2).                     |
| `datetime`        | Date and time the image was taken (extracted from EXIF data).                |
| `embedding`       | Numerical vector representation of the image (from CLIP).                    |
| `ocr_text`        | Text detected in the image (from PaddleOCR).                                |


## 🧰 Technologies Used

| Component            | Technology                             |
|---------------------|-------------------------------------|
| Database              | PostgreSQL + pgvector                  |
| Image Captioning      | BLIP-2 (Hugging Face Transformers)      |
| Object Detection      | Detectron2                             |
| Image Embedding       | CLIP (OpenAI)                          |
| OCR                   | PaddleOCR                              |
| Clustering            | t-SNE + HDBSCAN                        |
| Metadata Management   | SQLAlchemy                             |



## ✅ Best Practices

- **Batch Processing**: Process images in batches to optimize resource usage and minimize model loading overhead.
- **Regular Clustering**: Periodically rerun clustering (t-SNE + HDBSCAN) to group newly added images into albums.
- **Vector Search for Recommendations**: Use `pgvector` similarity search to quickly find similar images based on embeddings.
- **Centralized Metadata Storage**: Store all extracted metadata (captions, objects, OCR text, embeddings) in PostgreSQL for easy querying and analysis.
- **Flexible Device Configuration**: Set `device.default_device` in `config.yaml` to switch between GPU (CUDA), Apple Silicon (MPS), or CPU based on your hardware.

---

## 🚨 Planned Enhancements

- **Perceptual Hashing (pHash)**: Implement duplicate image detection using perceptual hashing to automatically detect and skip duplicate images.
- **Web-based UI**: Develop a simple browser-based interface to browse albums, search images, and view metadata.
- **Generative Captions and Summaries**: Enhance album summaries using LLMs to generate richer, more personalized album descriptions.
- **Distributed Processing**: Enable support for processing large-scale photo libraries across multiple machines.
- **Cloud Storage Integration**: Add support to ingest images directly from cloud storage providers (S3, GCS, etc.).

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for full details.

---

## 👤 Author

**Muhammad Mamduh Ahmad Zabidi**
Senior Data Engineer | Computational Biology Specialist

---

## 📬 Contact

For feedback, contributions, or collaboration opportunities, feel free to reach out.
