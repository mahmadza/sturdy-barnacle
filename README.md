# Sturdy Barnacle - ML-Powered Image Management System

**Sturdy Barnacle** is an image management system that extracts metadata, detects objects, generates embeddings, and organizes images into albums using machine learning techniques.

## Features
- **Image Metadata Extraction**: Extracts EXIF metadata from images.
- **Object Detection**: Identifies objects in images using Detectron2.
- **Image Captioning**: Generates descriptive captions with BLIP-2.
- **Image Embedding & Similarity Search**: Uses CLIP embeddings to find similar images.
- **Automatic Album Organization**: Groups similar images into albums using t-SNE & HDBSCAN.
- **Auto-Naming & Tagging Albums**: Generates meaningful names and tags for albums.

## Installation

### **1️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **2️⃣ Configure Database**
Modify `config.yaml` to set up the PostgreSQL database URL.

### **3️⃣ Initialize Database**
```sh
python -c "from sturdy_barnacle.db_utils import DatabaseManager; DatabaseManager()._initialize_db()"
```

### **4️⃣ Run Image Processing**
Refer to examples in `notebooks/` directory.

### **5️⃣ Auto-Organize Albums**
Refer to examples in `notebooks/` directory.


### **6️⃣ Run Tests**
(to be added)

## Project Structure
```
├── sturdy_barnacle
│   ├── db_utils.py           # Database connection & utilities
│   ├── image_processor.py    # Image processing & metadata extraction
│   ├── image_album_manager.py # Clustering images into albums
│   ├── auto_album_manager.py # Album naming, tagging, & summaries
│   ├── image_visualizer.py   # Visualization of images & metadata
├── notebooks                 # example notebooks
├── tests
├── requirements.txt          # Dependencies
├── config.yaml               # Configuration file
└── README.md
```


## Contributing
Feel free to fork, open issues, or submit pull requests.

## License
This project is licensed under the MIT License.
