from sklearn.neighbors import NearestNeighbors
import pandas as pd

class KNNRecommendationEngine:
    def __init__(self, data_model):
        self.data_model = data_model
        # We will fit the NearestNeighbors model after filtering by genre
        self.neigh = NearestNeighbors()

    def generate_recommendations(self, test_feat, playlist_size, selected_genre):
        genre_filtered_data = self.data_model.global_dataset[
            self.data_model.global_dataset['artist_genres'].apply(
                lambda genres: selected_genre.lower() in [g.lower() for g in genres]
            )
        ]

        if genre_filtered_data.empty:
            return pd.DataFrame(columns=self.data_model.global_dataset.columns)

        self.neigh.fit(genre_filtered_data[self.data_model.audio_features].to_numpy())

        distances, indices = self.neigh.kneighbors([test_feat], n_neighbors=playlist_size)
        indices = indices.flatten()
        recommendations = genre_filtered_data.iloc[indices]
        return recommendations
