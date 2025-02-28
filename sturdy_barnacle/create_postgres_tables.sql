
CREATE DATABASE images_db;

\c images_db;

CREATE EXTENSION vector;

CREATE TABLE IF NOT EXISTS image_metadata (
    id SERIAL PRIMARY KEY,
    image_path TEXT UNIQUE NOT NULL,
    description TEXT,
    detected_objects TEXT,
    datetime TEXT,
    embedding VECTOR(512),
    ocr_text TEXT
);

CREATE TABLE IF NOT EXISTS image_albums (
    id SERIAL PRIMARY KEY,
    album_name TEXT UNIQUE NOT NULL
);


CREATE TABLE IF NOT EXISTS image_album_mapping (
    album_id INTEGER REFERENCES image_albums(id) ON DELETE CASCADE,
    image_path TEXT REFERENCES image_metadata(image_path) ON DELETE CASCADE,
    PRIMARY KEY (album_id, image_path)
);


GRANT CONNECT ON DATABASE images_db TO myuser;
GRANT USAGE ON SCHEMA public TO myuser;

GRANT USAGE, SELECT, UPDATE ON SEQUENCE public.image_metadata_id_seq TO myuser;


GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO myuser;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO myuser;
