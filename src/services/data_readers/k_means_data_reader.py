import pandas as pd
from src.models.k_means_clustering_recommender_data_model import KMeansClusteringRecommenderDataModel
from ast import literal_eval


class KMeansDataReaderPkl:
    def __init__(self, path_to_features_dataset_pkl):
        self._path_to_features_dataset_pkl = path_to_features_dataset_pkl
        self.__df = pd.read_pickle(self._path_to_features_dataset_pkl)

    def get_data_model(self):
        model_data: pd.DataFrame = self.__df.drop(columns=['album_id', 'album_name', 'album_type', 'artists',
                                                           'release_date', 'track_id', 'artist_genres', 'track_name',
                                                           'album_popularity', 'artist_popularity', 'explicit', 'time_signature',
                                                           'key', 'track_popularity', 'mode', 'duration_ms'], axis=1).copy()

        genres_df = self.__df['artist_genres'].apply(literal_eval).copy().to_frame().explode("artist_genres")
        artists_df = self.__df['artists'].apply(literal_eval).copy().to_frame().explode("artists")

        return KMeansClusteringRecommenderDataModel(self.__df, model_data, genres_df, artists_df)

    @staticmethod
    def get_model_features(df: pd.DataFrame):
        return df.drop(columns=['album_id', 'album_name', 'album_type', 'artists',
                                'release_date', 'track_id', 'artist_genres', 'track_name',
                                'album_popularity', 'artist_popularity', 'explicit', 'time_signature',
                                'key', 'track_popularity', 'mode', 'duration_ms'], axis=1)

