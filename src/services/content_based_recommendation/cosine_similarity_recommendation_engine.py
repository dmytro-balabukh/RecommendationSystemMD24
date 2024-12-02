from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy as sc


class CosineSimilarityRecommendationEngine:
    def __init__(self, global_features_sparse, track_info):
        self._global_features_sparse = global_features_sparse
        self._track_info = track_info.values

    def generate_recommendations(self, client_feature_vector, client_features_indices, playlist_size, exclude_artists_from_client_playlist):
        cosine_similarities = cosine_similarity(np.asarray(client_feature_vector),
                                                sc.sparse.csr_matrix.toarray(self._global_features_sparse)).flatten()
        # Exclude client's tracks from recommendations by setting their similarity to -inf
        cosine_similarities[client_features_indices] = -np.inf
        top_indices = np.argsort(-cosine_similarities)[:playlist_size]
        top_recommendations = self._track_info[top_indices]

        return top_recommendations
