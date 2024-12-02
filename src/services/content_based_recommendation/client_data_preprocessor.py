# TODO: Update comments
import pandas as pd
from pandas import Series


class ClientDataPreProcessor:
    def __init__(self, global_features_df):
        self._global_features_df = global_features_df

    def generate_client_features_vector(self, client_features_df: pd.DataFrame) -> pd.Series:
        """
        Summarize a user's playlist into a single vector.
        ---
        Input:
        global_features_df (pandas dataframe): Dataframe which includes all the features for the spotify songs
        playlist_df (pandas dataframe): playlist dataframe

        Output:
        global_features_df_playlist_final (pandas series): single vector feature that summarizes the playlist
        """

        # Find song features in the playlist
        global_features_df_playlist = self._global_features_df[
            self._global_features_df['track_id'].isin(client_features_df['track_id'].values)]
        # Drop columns that won't be used in calculations
        global_features_df_playlist_final = global_features_df_playlist.drop(
            columns=['artists', 'release_date', 'track_id', 'album_name', 'track_name', 'track_href'])
        return global_features_df_playlist_final.sum(axis=0)
