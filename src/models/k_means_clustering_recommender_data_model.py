class KMeansClusteringRecommenderDataModel:
    def __init__(self, global_dataset, model_data, genres_df, artists_df):
        self.global_dataset = global_dataset
        self.model_data = model_data
        self.model_data_scaled = None
        self.genres_df = genres_df
        self.artists_df = artists_df
