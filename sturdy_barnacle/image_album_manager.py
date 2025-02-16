from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import yaml
from sklearn.manifold import TSNE
from sqlalchemy import text

from sturdy_barnacle.db_utils import TABLES, DatabaseManager

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


class ImageAlbumManager:
    """Class to group images into albums using t-SNE + HDBSCAN."""

    def __init__(
        self, db: DatabaseManager, min_cluster_size=10, min_samples=2
    ):
        self.db = db
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def load_embeddings(self):
        """Fetch image embeddings from the database."""
        images = self.db.query_all_images()
        embeddings = np.array([img.embedding for img in images])
        return embeddings, images

    def apply_tsne(self, embeddings, n_components=2, perplexity=30.0):
        """Reduces dimensions using t-SNE for better clustering."""
        reducer = TSNE(
            n_components=n_components, perplexity=perplexity, random_state=42
        )
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings

    def cluster_hdbscan(self, embeddings):
        """Clusters using HDBSCAN."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        return labels

    def group_images_into_albums(self):

        self.reset_album_tables()

        embeddings, images = self.load_embeddings()

        tsne_embeddings = self.apply_tsne(embeddings)

        labels = self.cluster_hdbscan(tsne_embeddings)

        album_data = []
        album_ids = {}
        unique_clusters = set(labels) - {-1}  # Ignore noise (-1)

        for cluster in unique_clusters:
            album_name = f"Album {cluster+1}"
            album_id = self.db.create_album(album_name)
            album_ids[cluster] = album_id

        image_counts = {album_id: 0 for album_id in album_ids.values()}

        for i, img in enumerate(images):
            cluster_label = labels[i]
            if cluster_label != -1:
                album_id = album_ids[cluster_label]
                self.db.add_image_to_album(album_id, img.image_path)
                image_counts[album_id] += 1

        for album_id, num_images in image_counts.items():
            album_data.append(
                {
                    "album_id": album_id,
                    "album_name": f"Album {album_id}",
                    "num_images": num_images,
                }
            )

        df = pd.DataFrame(album_data).sort_values(
            by="num_images", ascending=False
        )

        print("Albums created successfully using t-SNE + HDBSCAN!")
        return df

    def reset_album_tables(self):
        """Clears the album tables before re-clustering."""
        session = self.db._get_session()
        try:
            album_mapping_table = TABLES.get("image_album_mapping")
            albums_table = TABLES.get("image_albums")

            if not album_mapping_table or not albums_table:
                raise ValueError("Invalid table name detected!")

            session.execute(
                text(
                    f"""
                    TRUNCATE TABLE {album_mapping_table}
                    RESTART IDENTITY CASCADE;
                """
                )
            )
            session.execute(
                text(
                    f"""
                    TRUNCATE TABLE {albums_table} RESTART IDENTITY CASCADE;
                """
                )
            )
            session.commit()
            print("Tables cleared. Ready for a new clustering experiment.")
        finally:
            session.close()
