import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.models.cosine_similarity_recommender_data_model import CosineSimilarityRecommenderDataModel
class CosineSimilarityDataReaderPkl:
    def __init__(self, path_to_features_dataset_pkl):
        self._path_to_features_dataset_pkl = path_to_features_dataset_pkl
        self.__df = pd.read_pickle(self._path_to_features_dataset_pkl)

    def get_global_features_and_metadata(self):
        numeric_features_with_genres = self.__df.drop(
            columns=['artists', 'release_date', 'track_id', 'album_name', 'track_name', 'track_href']).astype(float)

        #non_genre_columns = self.__get_non_genre_columns_names()
        complete_dataset_without_genres = pd.read_pickle("C:\\Users\\dbala\\Documents\\repos\\University\\Data\\content_based_kmeans_dataset.pkl")
        return CosineSimilarityRecommenderDataModel(csr_matrix(numeric_features_with_genres.values), complete_dataset_without_genres, self.__get_feature_indices())

    def __get_feature_indices(self):
        return {name: index for index, name in enumerate(self.__df.columns)}

    def __get_non_genre_columns_names(self):
        return [column_name for column_name in self.__df.columns if "genre|" not in column_name]
