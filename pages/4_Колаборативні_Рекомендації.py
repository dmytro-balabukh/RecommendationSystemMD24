# user_history_recommendations.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Import your existing helper functions and classes
from pages.helpers.streamlit_ui_helper import StreamlitUiHelper

class UserHistoryRecommendationsStreamlitPage:
    def __init__(self):
        self.init_content()
        self.init_sidebar()
        StreamlitUiHelper.hide_deploy_button()

    @staticmethod
    def init_content():
        st.markdown("# Рекомендації на основі історії прослуховувань")
        st.markdown("#### Завантажте вашу історію прослуховувань у форматі JSON для отримання персоналізованих рекомендацій.")

        with st.expander("Детальніше про рекомендації на основі колаборативної фільтрації"):
            st.markdown("""
            Тут можна додати опис того, як працюють рекомендації на основі колаборативної фільтрації.
            """)

    def init_sidebar(self):
        st.sidebar.header("Завантаження історії прослуховувань")
        uploaded_file = st.sidebar.file_uploader("Завантажте файл JSON з історією прослуховувань", type="json")

        if uploaded_file is not None:
            with st.spinner("Обробка вашої історії прослуховувань..."):
                self.process_user_history(uploaded_file)

    def process_user_history(self, uploaded_file):
        # Load user's listening history
        user_history = pd.read_json(uploaded_file)

        # Preprocess user's listening history
        # Define a threshold for a valid play (e.g., 30 seconds)
        VALID_PLAY_THRESHOLD = 30000  # in milliseconds

        # Create a 'plays' column based on 'msPlayed'
        user_history['plays'] = user_history['msPlayed'].apply(lambda x: 1 if x >= VALID_PLAY_THRESHOLD else 0)

        # Aggregate plays per track
        user_track_plays = user_history.groupby(['artistName', 'trackName'])['plays'].sum().reset_index()

        # Load mappings and data
        user_item_matrix = self.load_pickle_file('user_item_matrix.pkl')
        user_track_interactions = self.load_pickle_file('user_track_interactions.pkl')
        track_id_map = self.load_pickle_file('track_id_map.pkl')
        track_id_to_idx = self.load_pickle_file('track_id_to_idx.pkl')
        track_idx_to_id = self.load_pickle_file('track_idx_to_id.pkl')
        user_id_map = self.load_pickle_file('user_id_map.pkl')
        user_id_to_idx = self.load_pickle_file('user_id_to_idx.pkl')

        # Map user's tracks to your dataset
        # Create a DataFrame of unique tracks in your dataset
        unique_tracks = pd.DataFrame(list(track_id_map.items()), columns=['track_enc', 'track'])
        unique_tracks['track_enc'] = unique_tracks['track_enc'].astype(int)

        # Merge on 'trackName'
        user_track_plays = user_track_plays.merge(
            unique_tracks,
            left_on='trackName',
            right_on='track',
            how='left'
        )

        # Drop unmatched tracks
        user_track_plays = user_track_plays.dropna(subset=['track_enc'])
        user_track_plays['track_enc'] = user_track_plays['track_enc'].astype(int)

        if user_track_plays.empty:
            st.error("Не вдалося знайти відповідності для ваших треків у нашій базі даних.")
            return

        # Assign a new user_id_enc
        new_user_id_enc = max(user_id_map.keys()) + 1

        # Create new user interactions DataFrame
        new_user_interactions = pd.DataFrame({
            'user_id_enc': new_user_id_enc,
            'track_enc': user_track_plays['track_enc'],
            'freq': user_track_plays['plays']
        })

        # Combine with existing interactions
        user_track_interactions = pd.concat(
            [user_track_interactions, new_user_interactions],
            ignore_index=True
        )

        # Update user_id_map and user_id_to_idx
        user_id_map[new_user_id_enc] = 'new_user'
        user_id_to_idx[new_user_id_enc] = len(user_id_to_idx)

        # Map IDs in the interactions DataFrame to indices
        user_indices = user_track_interactions['user_id_enc'].map(user_id_to_idx)
        track_indices = user_track_interactions['track_enc'].map(track_id_to_idx)
        frequencies = user_track_interactions['freq'].astype(float)

        # Recreate the sparse user-item matrix
        user_item_matrix = csr_matrix(
            (frequencies, (user_indices, track_indices)),
            shape=(len(user_id_to_idx), len(track_id_to_idx))
        )

        # Perform SVD
        n_components = 20  # Number of latent factors
        SVD = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = SVD.fit_transform(user_item_matrix)
        item_factors = SVD.components_.T

        # Generate recommendations
        recommendations = self.generate_recommendations(
            new_user_id_enc,
            user_factors,
            item_factors,
            user_id_to_idx,
            track_id_to_idx,  # Pass track_id_to_idx
            track_idx_to_id,
            track_id_map,
            user_track_interactions
        )

        # Display recommendations
        st.write('### Рекомендації треків для вас:')
        for idx, track_name in enumerate(recommendations, 1):
            st.write(f"{idx}. {track_name}")

    @staticmethod
    def generate_recommendations(
        user_id_enc,
        user_factors,
        item_factors,
        user_id_to_idx,
        track_id_to_idx,  # Add this parameter
        track_idx_to_id,
        track_id_map,
        user_track_interactions,
        n_recommendations=10
    ):
        # Get the index of the new user
        user_idx = user_id_to_idx.get(user_id_enc)
        if user_idx is None:
            print("User ID not found in the user index mapping.")
            return []

        # Get the user's latent factors
        user_vector = user_factors[user_idx]

        # Compute scores (dot product between user vector and item factors)
        scores = item_factors.dot(user_vector)

        # Get indices of tracks the user has already interacted with
        interacted_tracks = user_track_interactions[user_track_interactions['user_id_enc'] == user_id_enc]['track_enc']
        interacted_track_indices = [track_id_to_idx[track_id] for track_id in interacted_tracks]

        # Exclude already interacted tracks
        scores[interacted_track_indices] = -np.inf

        # Get top recommendations
        top_item_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommended_track_encs = [track_idx_to_id[idx] for idx in top_item_indices]
        recommended_tracks = [track_id_map[enc_id] for enc_id in recommended_track_encs]

        return recommended_tracks

    @staticmethod
    def load_pickle_file(filename):
        with open(f'C:\\Users\\dbala\\Documents\\repos\\University\\Data\\collaborative_based_data\\{filename}', 'rb') as f:
            return pickle.load(f)

# Main
st.set_page_config(page_title="User History Recommendations", page_icon="🎧")
page = UserHistoryRecommendationsStreamlitPage()
