import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import hdbscan
import numpy as np
import pandas as pd
import yaml
from nltk.corpus import wordnet
from sklearn.manifold import TSNE
from sqlalchemy import text

from sturdy_barnacle.db_utils import TABLES, DatabaseManager

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


class AlbumManager:
    """Manages album creation, clustering, and metadata enrichment."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.tsne_params = CONFIG["models"]["tsne"]
        self.hdbscan_params = CONFIG["models"]["hdbscan"]

    def group_images_into_albums(self) -> pd.DataFrame:
        """Clusters images into albums using t-SNE + HDBSCAN and resets tables."""

        self._reset_album_tables()

        embeddings, images = self._load_image_embeddings()

        if len(embeddings) == 0:
            print("No images found with embeddings. Skipping clustering.")
            return pd.DataFrame()

        reduced_embeddings = self._apply_tsne(embeddings)

        labels = self._cluster_with_hdbscan(reduced_embeddings)

        album_ids = self._create_albums_for_clusters(labels)

        self._assign_images_to_albums(images, labels, album_ids)

        df = self._generate_album_summary_df(album_ids)

        print("Albums created successfully using t-SNE + HDBSCAN!")
        return df

    def _reset_album_tables(self) -> None:
        """Clears all existing album data."""
        session = self.db._get_session()
        try:
            album_mapping_table = TABLES.get("image_album_mapping")
            albums_table = TABLES.get("image_albums")

            if not album_mapping_table or not albums_table:
                raise ValueError("Invalid table names in config.")

            session.execute(
                text(
                    f"TRUNCATE TABLE {album_mapping_table} RESTART IDENTITY CASCADE;"
                )
            )
            session.execute(
                text(
                    f"TRUNCATE TABLE {albums_table} RESTART IDENTITY CASCADE;"
                )
            )
            session.commit()
            print("Album tables reset.")
        finally:
            session.close()

    def _load_image_embeddings(self) -> Tuple[np.ndarray, List]:
        """Loads all images with embeddings from the database."""
        images = self.db.query_all_images()
        embeddings = [img.embedding for img in images if img.embedding]
        return np.array(embeddings), images

    def _apply_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduces embedding dimensions using t-SNE."""
        tsne = TSNE(
            n_components=self.tsne_params["n_components"],
            perplexity=self.tsne_params["perplexity"],
            random_state=42,
        )
        return tsne.fit_transform(embeddings)

    def _cluster_with_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """Clusters images using HDBSCAN."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_params["min_cluster_size"],
            min_samples=self.hdbscan_params["min_samples"],
            metric="euclidean",
        )
        return clusterer.fit_predict(embeddings)

    def _create_albums_for_clusters(
        self, labels: np.ndarray
    ) -> Dict[int, int]:
        """Creates albums for each cluster and returns cluster-to-album ID map."""
        unique_clusters = set(labels) - {-1}  # Exclude noise cluster (-1)
        album_ids = {}
        for cluster in unique_clusters:
            album_name = f"Album {cluster+1}"
            album_id = self.db.create_album(album_name)
            album_ids[cluster] = album_id
        return album_ids

    def _assign_images_to_albums(self, images, labels, album_ids) -> None:
        """Links images to albums based on cluster labels."""
        for i, img in enumerate(images):
            cluster_label = labels[i]
            if cluster_label != -1:  # Ignore noise
                album_id = album_ids[cluster_label]
                self.db.add_image_to_album(album_id, img.image_path)

    def _generate_album_summary_df(
        self, album_ids: Dict[int, int]
    ) -> pd.DataFrame:
        """Generates a summary DataFrame for all albums."""
        album_data = []
        for album_id in album_ids.values():
            image_count = len(self.db.get_album_images(album_id))
            album_data.append(
                {
                    "album_id": album_id,
                    "album_name": f"Album {album_id}",
                    "num_images": image_count,
                }
            )

        return pd.DataFrame(album_data).sort_values(
            by="num_images", ascending=False
        )

    # ---- Auto-naming, tagging, and summarization ----

    def auto_name_and_tag_albums(self) -> None:
        """Processes all albums and auto-generates names, tags, and summaries."""

        session = self.db._get_session()
        try:
            album_ids = session.execute(
                text("SELECT id FROM image_albums")
            ).fetchall()
            for (album_id,) in album_ids:
                detected_objects, descriptions, exif_locations = (
                    self._gather_album_metadata(album_id)
                )

                album_name = self._generate_album_name(
                    detected_objects, descriptions
                )
                tags = json.dumps(
                    self._generate_album_tags(
                        detected_objects, descriptions, exif_locations
                    )
                )
                summary = self._generate_album_summary(
                    detected_objects, descriptions, exif_locations
                )

                session.execute(
                    text(
                        f"""
                        UPDATE {TABLES.get("image_albums")}
                        SET album_name = :name, tags = :tags, summary = :summary
                        WHERE id = :album_id
                        """
                    ),
                    {
                        "name": album_name,
                        "tags": tags,
                        "summary": summary,
                        "album_id": album_id,
                    },
                )
                session.commit()

            print("Albums auto-named, tagged, and summarized.")
        finally:
            session.close()

    def _gather_album_metadata(
        self, album_id: int
    ) -> Tuple[Counter, List[str], List[str]]:
        """Gathers metadata for all images in a given album."""
        images = [
            self.db.get_image_by_path(path)
            for path in self.db.get_album_images(album_id)
        ]

        detected_objects = Counter()
        descriptions = []
        exif_locations = []

        for img in images:
            if img.detected_objects:
                detected_objects.update(json.loads(img.detected_objects))
            if img.description:
                descriptions.append(img.description)
            if img.exif_data and "GPSInfo" in img.exif_data:
                exif_locations.append(str(img.exif_data["GPSInfo"]))

        return detected_objects, descriptions, exif_locations

    def _generate_album_name(
        self, detected_objects: Counter, descriptions: List[str]
    ) -> str:
        """Generates album name based on most common object or first description."""
        if detected_objects:
            return f"{detected_objects.most_common(1)[0][0]} Collection"
        if descriptions:
            return f"{descriptions[0].split()[0]} Memories"
        return "Untitled Album"

    def _generate_album_tags(
        self,
        detected_objects: Counter,
        descriptions: List[str],
        exif_locations: List[str],
    ) -> List[str]:
        """Generates tags based on detected objects, descriptions, and locations."""
        tags = list(detected_objects.keys())
        tags.extend(self._extract_keywords_from_descriptions(descriptions))
        if exif_locations:
            tags.append("Location-based")
        return list(set(tags))

    def _extract_keywords_from_descriptions(
        self, descriptions: List[str]
    ) -> List[str]:
        """Extracts and merges keywords from descriptions."""
        keywords = [
            word
            for desc in descriptions
            for word in re.findall(r"\b[A-Za-z]{3,}\b", desc)
        ]
        keyword_counts = Counter(keywords).most_common(10)
        return self._merge_synonyms([kw[0] for kw in keyword_counts])

    def _merge_synonyms(self, keywords: List[str]) -> List[str]:
        """Merges synonyms to avoid redundant tags."""
        synonym_map = {}
        for word in keywords:
            synonyms = {
                lemma.name().replace("_", " ")
                for syn in wordnet.synsets(word)
                for lemma in syn.lemmas()
            }
            base_word = min(synonyms & set(keywords), key=len, default=word)
            synonym_map[word] = base_word
        return list(set(synonym_map.values()))

    def _generate_album_summary(
        self,
        detected_objects: Counter,
        descriptions: List[str],
        exif_locations: List[str],
    ) -> str:
        """Generates a human-readable summary for the album."""
        summary = f"This album contains images with objects such as {', '.join(detected_objects.keys()[:5])}. "
        if descriptions:
            summary += f"Some descriptions: {', '.join(descriptions[:5])}. "
        if exif_locations:
            summary += f"Locations: {', '.join(exif_locations)}."
        return summary.strip()
