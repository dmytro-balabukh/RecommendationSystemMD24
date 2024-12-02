import pandas as pd
from src.models.knn_recommender_data_model import KNNRecommenderDataModel
from ast import literal_eval

class KNNDataReaderPkl:
    def __init__(self, path_to_features_dataset_pkl):
        self._path_to_features_dataset_pkl = path_to_features_dataset_pkl
        self.__df = pd.read_pickle(self._path_to_features_dataset_pkl)

    def get_data_model(self):
        audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]
        model_data = self.__df.copy()
        model_data = model_data.dropna(subset=audio_feats)
        if 'tempo' not in model_data.columns:
            model_data['tempo'] = 0.0
        if 'artists' in model_data.columns:
            model_data['artists'] = model_data['artists'].apply(literal_eval)
        if 'artist_genres' in model_data.columns:
            model_data['artist_genres'] = model_data['artist_genres'].apply(literal_eval)
        return KNNRecommenderDataModel(model_data, audio_feats)

