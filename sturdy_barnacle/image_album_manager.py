from typing import Dict
import numpy as np
import hdbscan
from sturdy_barnacle.db_utils import DatabaseManager


class ImageAlbumManager:

    def __init__(self, db: DatabaseManager):
        self.db = DatabaseManager()

    def group_images_into_albums(self, min_cluster_size=5):
        """Clusters images using HDBSCAN and stores them as albums in the database."""
        images = self.db.query_all_images()
        if len(images) < min_cluster_size:
            print("Not enough images to form meaningful clusters.")
            return []

        embeddings = np.array([img.embedding for img in images])

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)

        album_ids = {}
        unique_clusters = set(labels) - {-1}  # Exclude noise (-1)

        for cluster in unique_clusters:
            album_name = f"Album {cluster+1}"  # Album names are auto-generated
            album_id = self.db.create_album(album_name)
            album_ids[cluster] = album_id

        for i, img in enumerate(images):
            cluster_label = labels[i]
            if cluster_label != -1:  # Ignore noise points
                album_id = album_ids[cluster_label]
                self.db.add_image_to_album(album_id, img.image_path)

        print("Albums created successfully!")
        return album_ids