from src.models.k_means_clustering_recommender_data_model import KMeansClusteringRecommenderDataModel


class KMeansDataPreprocessor:
    def __init__(self, data_model: KMeansClusteringRecommenderDataModel):
        self._data_model = data_model

    def preprocess_model_dataset(self, scaler):
        self._data_model.model_data_scaled = scaler.fit_transform(self._data_model.model_data)
