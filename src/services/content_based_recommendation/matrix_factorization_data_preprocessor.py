import pandas as pd
import scipy.sparse as sp

class MatrixFactorizationDataPreprocessor:
    def __init__(self, user_artists_path, artists_path):
        self.user_artists_path = user_artists_path
        self.artists_path = artists_path
        self.user_artists = None
        self.artists = None
        self.user_item_matrix = None
        self.user_id_map = None
        self.artist_id_map = None
        self.artist_id_to_index = None
        self.index_to_artist_id = None

    def load_data(self):
        self.user_artists = pd.read_csv(self.user_artists_path, sep="\t")
        self.artists = pd.read_csv(self.artists_path, sep="\t")

    def preprocess_data(self):
        user_ids = self.user_artists.userID.unique()
        artist_ids = self.user_artists.artistID.unique()
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.artist_id_map = {artist_id: idx for idx, artist_id in enumerate(artist_ids)}
        self.artist_id_to_index = self.artist_id_map
        self.index_to_artist_id = {idx: artist_id for artist_id, idx in self.artist_id_map.items()}

        row_indices = self.user_artists['userID'].map(self.user_id_map)
        col_indices = self.user_artists['artistID'].map(self.artist_id_map)
        data = self.user_artists['weight'].astype(float)
        self.user_item_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_id_map), len(self.artist_id_map))
        )

    def get_artist_name_by_id(self, artist_id):
        return self.artists.loc[self.artists['id'] == artist_id, 'name'].values[0]
