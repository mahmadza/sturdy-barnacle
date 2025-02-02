import json
import re
from collections import Counter

from nltk.corpus import wordnet
from sqlalchemy import text

from sturdy_barnacle.db_utils import DatabaseManager


class AutoAlbumManager:
    """Class to generate album names, tags, and summaries."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def get_cluster_metadata(self, album_id: int):
        """Fetch all images from an album and analyze metadata."""
        image_paths = self.db.get_album_images(album_id)
        metadata_list = [
            self.db.get_image_by_path(path) for path in image_paths
        ]

        detected_objects = Counter()
        descriptions = []
        exif_locations = []

        for metadata in metadata_list:
            if metadata.detected_objects:
                detected_objects.update(json.loads(metadata.detected_objects))
            if metadata.description:
                descriptions.append(metadata.description)
            if metadata.exif_data and "GPSInfo" in metadata.exif_data:
                exif_locations.append(metadata.exif_data["GPSInfo"])

        return detected_objects, descriptions, exif_locations

    def extract_keywords_from_description(self, descriptions):
        """Extracts meaningful keywords
        from descriptions and merges synonyms."""
        keywords = []
        for desc in descriptions:
            words = re.findall(
                r"\b[A-Za-z]{3,}\b", desc
            )  # Extract words of 3+ letters
            keywords.extend(words)

        keyword_counts = Counter(keywords).most_common(
            10
        )  # Get top 10 keywords
        return self.merge_synonyms([kw[0] for kw in keyword_counts])

    def merge_synonyms(self, keywords):
        """Merges synonyms to avoid redundant tags."""
        synonym_map = {}
        for word in keywords:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemas():
                    synonyms.add(lemma.name().replace("_", " "))

            synonyms = synonyms.intersection(
                set(keywords)
            )  # Keep only relevant synonyms
            base_word = min(synonyms, key=len) if synonyms else word
            synonym_map[word] = base_word

        return list(set(synonym_map.values()))

    def generate_album_name(self, detected_objects, descriptions):
        """Generate an album name based on objects and descriptions."""
        if detected_objects:
            common_object = detected_objects.most_common(1)[0][0]
            return f"{common_object.capitalize()} Collection"
        elif descriptions:
            return descriptions[0].split()[0] + " Memories"
        return "Untitled Album"

    def generate_tags(self, detected_objects, descriptions, exif_locations):
        """Generate a list of tags based on objects,
        descriptions, and locations with synonyms merged."""
        tags = list(detected_objects.keys())
        desc_keywords = self.extract_keywords_from_description(descriptions)
        tags.extend(desc_keywords)
        if exif_locations:
            tags.append("Location-based")
        return list(set(tags))  # Remove duplicates

    def generate_album_summary(
        self, detected_objects, descriptions, exif_locations
    ):
        """Generate a simple album summary based on detected elements."""
        summary = f"This album contains images with objects \
            such as {', '.join(detected_objects.keys()[:5])}. "
        if descriptions:
            summary += (
                f"Some descriptions include: {', '.join(descriptions[:5])}. "
            )
        if exif_locations:
            summary += f"These images were taken at \
                locations: {', '.join(map(str, exif_locations))}."
        return summary.strip()

    def auto_name_and_tag_albums(self):
        """Process all albums and update their names, tags, and summaries."""
        session = self.db._get_session()
        try:
            album_ids = session.execute(text("SELECT id FROM image_albums"))
            for album_id in album_ids:
                detected_objects, descriptions, exif_locations = (
                    self.get_cluster_metadata(album_id[0])
                )
                album_name = self.generate_album_name(
                    detected_objects, descriptions
                )
                tags = json.dumps(
                    self.generate_tags(
                        detected_objects, descriptions, exif_locations
                    )
                )
                summary = self.generate_album_summary(
                    detected_objects, descriptions, exif_locations
                )

                session.execute(
                    text(
                        """
                        UPDATE image_albums
                        SET album_name = :name,
                        tags = :tags, summary = :summary
                        WHERE id = :album_id
                    """
                    ),
                    {
                        "name": album_name,
                        "tags": tags,
                        "summary": summary,
                        "album_id": album_id[0],
                    },
                )
                session.commit()
            print("✅ Albums updated with basic summaries!")
        finally:
            session.close()
