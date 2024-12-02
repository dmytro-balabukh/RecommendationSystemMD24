import pandas as pd
from scipy.sparse import csr_matrix

import numpy as np
from src.services.content_based_recommendation.client_data_preprocessor import ClientDataPreProcessor
from src.services.content_based_recommendation.cosine_similarity_recommendation_engine import CosineSimilarityRecommendationEngine
from src.services.data_readers.cosine_similarity_data_reader import CosineSimilarityDataReaderPkl

if __name__ == '__main__':
    pass
    reader = CosineSimilarityDataReaderPkl(
        "C:\\Users\dbala\Documents\\repos\\University\\Data\\content_based_374349_songs_complete.pkl")
    features_sparse, track_info = reader.get_global_features_and_metadata()
