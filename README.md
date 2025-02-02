# Sturdy Barnacle - AI-Powered Image Management System

**Sturdy Barnacle** is an advanced image management system that extracts metadata, detects objects, generates embeddings, and organizes images into albums using machine learning techniques.

## Features
- **Image Metadata Extraction**: Extracts EXIF metadata from images.
- **Object Detection**: Identifies objects in images using Detectron2.
- **Image Captioning**: Generates descriptive captions with BLIP-2.
- **Image Embedding & Similarity Search**: Uses CLIP embeddings to find similar images.
- **Automatic Album Organization**: Groups similar images into albums using t-SNE & HDBSCAN.
- **Auto-Naming & Tagging Albums**: Generates meaningful names and tags for albums.
- **AI-Generated Summaries**: Provides a text summary of each album based on detected content.

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
To process an image and store its metadata:
```sh
python image_processor.py /path/to/image.jpg
```

### **5️⃣ Auto-Organize Albums**
```sh
python auto_album_manager.py
```

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
├── tests
├── requirements.txt          # Dependencies
├── config.yaml               # Configuration file
└── README.md
```


## Contributing
Feel free to fork, open issues, or submit pull requests. Let's build the best AI-powered image manager together!

## License
This project is licensed under the MIT License.
