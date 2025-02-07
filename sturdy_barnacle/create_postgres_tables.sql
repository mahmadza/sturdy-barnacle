
# Create the tables for the image album application
# The image_albums table will store the album names
CREATE TABLE image_albums (
    id SERIAL PRIMARY KEY,
    album_name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

# The image_album_mapping table will store the mapping between albums and images
CREATE TABLE image_album_mapping (
    id SERIAL PRIMARY KEY,
    album_id INTEGER REFERENCES image_albums(id) ON DELETE CASCADE,
    image_path TEXT UNIQUE NOT NULL
);


# To get the number of images in each album, use the following query:
SELECT
    ia.id AS album_id,
    ia.album_name,
    COUNT(iam.image_path) AS num_images
FROM image_albums ia
LEFT JOIN image_album_mapping iam ON ia.id = iam.album_id
GROUP BY ia.id, ia.album_name
ORDER BY num_images DESC;


# Add OCR text and search vector columns
ALTER TABLE image_metadata ADD COLUMN ocr_text TEXT;
ALTER TABLE image_metadata ADD COLUMN search_vector TSVECTOR;

# Add indices
CREATE INDEX idx_search_vector ON image_metadata USING GIN(search_vector);
CREATE INDEX idx_embedding ON image_metadata USING HNSW (embedding vector_l2_ops);

# Verify indices
SELECT * FROM pg_indexes WHERE tablename = 'image_metadata';

# Create index
CREATE INDEX IF NOT EXISTS idx_album ON image_album_mapping(album_id);
