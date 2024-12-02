class CosineSimilarityRecommenderDataModel:
    def __init__(self, dataset_numeric_sparse, dataset_without_genres, columns_map):
        self.dataset_numeric_sparse = dataset_numeric_sparse
        self.dataset_without_genres = dataset_without_genres
        self.columns_map = columns_map
