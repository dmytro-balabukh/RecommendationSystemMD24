# This is a sample Python script.
import pandas as pd
from scipy.sparse import csr_matrix

import numpy as np
from src.services.content_based_recommendation.client_data_preprocessor import ClientDataPreProcessor
from src.services.content_based_recommendation.cosine_similarity_recommendation_engine import CosineSimilarityRecommendationEngine
from src.services.data_readers.cosine_similarity_data_reader import CosineSimilarityDataReaderPkl

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
    reader = CosineSimilarityDataReaderPkl(
        "C:\\Users\dbala\Documents\\repos\\University\\Data\\content_based_374349_songs_complete.pkl")
    features_sparse, track_info = reader.get_global_features_and_metadata()

    # Filter for "Texas Flood" directly in the track_info for efficiency
    #texas_flood_indices = np.where(track_info['album_name'] == 'Texas Flood (Legacy Edition)')[0]

    # Create a sparse matrix for "Texas Flood" by averaging features (assuming mean for demonstration)
    #texas_flood_features_sparse = features_sparse[texas_flood_indices].mean(axis=0).reshape(1, -1)
    #generator = CosineSimilarityRecommendationEngine(features_sparse, track_info)
    #recommendations = generator.generate_recommendations(texas_flood_features_sparse, texas_flood_indices)

    #for track_id, track_name in recommendations:
    #    print(f"{track_id}: {track_name}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
