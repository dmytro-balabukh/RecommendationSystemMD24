import pandas as pd
import streamlit
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy as sc
from scipy.spatial.distance import cdist

from src.services.data_readers.k_means_data_reader import KMeansDataReaderPkl


class KMeansRecommendationEngine:
    def __init__(self):
        self._model_data = None

    def get_frequent_clusters(self, predictions, top_n):

        unique_values, frequency = np.unique(predictions, return_counts=True)
        cluster_num = min(top_n, unique_values.shape[0])

        # get most frequent clusters
        sorted_indexes = np.argsort(frequency)[::-1]

        frequent_clusters = unique_values[sorted_indexes]
        sorted_freq = frequency[sorted_indexes]
        freq_sum = np.sum(sorted_freq[:cluster_num])  # only take sum for cluster_num
        # To what percentage should cluster x be used for the recommendation?
        freq_perc = [(i / freq_sum) for i in sorted_freq[:cluster_num]]
        return cluster_num, frequent_clusters, freq_perc

    def generate_recommendations(self,
                                 client_df,
                                 global_dataset,
                                 model_data,
                                 scaler,
                                 model,
                                 rec_max=5,
                                 top_n=3):

        # transform data and make cluster predictions
        song_ids = client_df['track_id'].values.tolist()

        client_data_numeric_features = KMeansDataReaderPkl.get_model_features(client_df)
        print("Cient_DF", client_df)

        client_data_transformed = scaler.transform(client_data_numeric_features)
        predictions = model.predict(client_data_transformed)

        # determine the most frequent cluster classes from user input
        # dependent on the frequency it is decided how many recommendations come from the respective classes
        # e.g. 20 % of predictions is class 3: .2 * rec_max recommendations are done with user items with class prediction 3
        cluster_num, frequent_clusters, freq_perc = self.get_frequent_clusters(predictions, top_n)


        recs_id = pd.DataFrame(columns=['track_id', 'similarity'])
        #recs = pd.DataFrame(columns=model_data.columns)
        for i in range(cluster_num):
            # determine number of recommendations
            rec_num = round(freq_perc[i] * rec_max)
            cluster_number = frequent_clusters[i]

            # make mean vector out of songs from cluster - best idea?
            # find position of elements from cluster
            pos = np.where(predictions == cluster_number)[0]
            cluster_songs = client_data_transformed[pos, :]

            mean_song = np.mean(cluster_songs, axis=0)

            # make rec_num recommendations using cluster_num

            # calculate similarity
            similarity = cdist(np.reshape(mean_song, (1, -1)), model_data)


            # sort, but keep id in mind
            similarity_s = pd.Series(similarity.flatten(), name='similarity')
            similar_songs = pd.concat(
                [global_dataset['track_id'].reset_index(drop=True), similarity_s.reset_index(drop=True)], axis=1)

            # remove songs from user_songs list
            similar_songs = similar_songs[~(similar_songs['track_id'].isin(song_ids))]
            similar_songs = similar_songs.sort_values(by='similarity', ascending=True).reset_index(drop=True)

            recs_id = recs_id._append(similar_songs)

        recs_id = recs_id.reset_index(drop=True)

        return recs_id.loc[:rec_max - 1]