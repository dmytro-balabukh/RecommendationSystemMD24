import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from ast import literal_eval

from pages.helpers.streamlit_ui_helper import StreamlitUiHelper
from src import constants
from src.clients.spotify_client import SpotifyClient
from src.services.data_readers.knn_data_reader import KNNDataReaderPkl
from src.services.content_based_recommendation.knn_recommendation_engine import KNNRecommendationEngine

class KNearestNeighborsRecommendationsStreamlitPage:
    def __init__(self):
        self.playlist_size = 0
        self.__spotify_client = SpotifyClient(
            "02aca3b049a74aeba6a820a0ba7e0c50",
            "513bcf2584eb43518a808ad653dfa9b8"
        )
        self.init_content()
        self.init_sidebar()
        StreamlitUiHelper.hide_deploy_button()

    def init_content(self):
        st.markdown("# Рекомендації методом найближчих сусідів")
        st.markdown("#### З можливістю ручного налаштування аудіо параметрів та вибору жанру")

        with st.expander("Детальніше про метод найближчих сусідів"):
            path = Path(__file__).parent / "../src/markdowns/knn_detailed_description.md"
            st.markdown(open(path, encoding="utf8").read())

        self.init_settings()

    def init_settings(self):
        st.header("Параметри рекомендацій")

        # Emotion presets
        emotion_presets = {
            '1': {'acousticness': 0.2, 'danceability': 0.6, 'energy': 0.8, 'instrumentalness': 0.1, 'valence': 0.3, 'tempo': 140.0},
            '2': {'acousticness': 0.7, 'danceability': 0.3, 'energy': 0.2, 'instrumentalness': 0.5, 'valence': 0.1, 'tempo': 90.0},
            '3': {'acousticness': 0.4, 'danceability': 0.8, 'energy': 0.7, 'instrumentalness': 0.1, 'valence': 0.9, 'tempo': 120.0},
            '4': {'acousticness': 0.8, 'danceability': 0.2, 'energy': 0.2, 'instrumentalness': 0.3, 'valence': 0.1, 'tempo': 80.0},
            '5': {'acousticness': 0.4, 'danceability': 0.7, 'energy': 0.6, 'instrumentalness': 0.2, 'valence': 0.7, 'tempo': 130.0},
            'Нейтральний': {'acousticness': 0.5, 'danceability': 0.5, 'energy': 0.5, 'instrumentalness': 0.2, 'valence': 0.5, 'tempo': 110.0},
        }

        # Layout adjustments
        col1, col2 = st.columns((2, 0.7))

        with col2:
            # User selects an emotion
            selected_emotion = st.selectbox('Виберіть заготовку:', list(emotion_presets.keys()), index=list(emotion_presets.keys()).index('Нейтральний'))

            # User selects a genre using radio buttons
            selected_genre_display_value = st.radio('Виберіть жанр:', list(constants.GENRE_MAP.keys()), index=list(constants.GENRE_MAP.keys()).index('Поп'))
            selected_genre = constants.GENRE_MAP[selected_genre_display_value]

        with col1:
            # Set default values based on the selected emotion
            preset = emotion_presets[selected_emotion]
            acousticness = preset['acousticness']
            danceability = preset['danceability']
            energy = preset['energy']
            instrumentalness = preset['instrumentalness']
            valence = preset['valence']
            tempo = preset['tempo']

            # User inputs audio features
            st.markdown("### Налаштуйте аудіо характеристики:")
            self.acousticness = st.slider('Акустичність', 0.0, 1.0, acousticness)
            self.danceability = st.slider('Танцювальність', 0.0, 1.0, danceability)
            self.energy = st.slider('Енергійність', 0.0, 1.0, energy)
            self.instrumentalness = st.slider('Інструментальність', 0.0, 1.0, instrumentalness)
            self.valence = st.slider('Позитивність', 0.0, 1.0, valence)
            self.tempo = st.slider('Темп', 0.0, 244.0, tempo)

        self.selected_genre = selected_genre

    def init_sidebar(self):
        st.sidebar.header("Параметри пошуку")

        with st.sidebar.form(key='Form1'):
            self.playlist_size: int = st.slider('Розмір рекомендаційного плейлиста', 5, 20, 12, 1)
            submitted = st.form_submit_button(label='Знайти рекомендації 🔎')

        if submitted:
            with st.spinner("Генерація рекомендацій"):
                self.generate_recommendations()

    def generate_recommendations(self):
        # Get the user's desired features
        test_feat = [self.acousticness, self.danceability, self.energy, self.instrumentalness, self.valence, self.tempo]
        # Load data and model
        data_model = self.get_data_model()
        # Initialize recommendation engine
        engine = KNNRecommendationEngine(data_model)
        # Generate recommendations
        recommendations = engine.generate_recommendations(test_feat, self.playlist_size, self.selected_genre)
        # Display recommendations
        st.write('### Рекомендації пісень')
        st.dataframe(
            recommendations[
                ['track_name', 'album_name', 'artists', 'artist_genres',
                 'danceability', 'valence', 'acousticness', 'speechiness',
                 'liveness', 'tempo', 'energy', 'instrumentalness']
            ],
            column_config={
                "track_name": st.column_config.Column(label="Назва треку", width='medium'),
                "album_name": "Назва альбому",
                "artists": st.column_config.ListColumn(label="Виконавці"),
                "artist_genres": st.column_config.ListColumn(label="Жанри"),
                "danceability": st.column_config.ProgressColumn(
                    "Танцювальність",
                    min_value=0,
                    max_value=1,
                ),
                "valence": st.column_config.ProgressColumn(
                    "Позитивність",
                    min_value=0,
                    max_value=1,
                ),
                "acousticness": st.column_config.ProgressColumn(
                    "Акустичність",
                    min_value=0,
                    max_value=1,
                ),
                "speechiness": st.column_config.ProgressColumn(
                    "Мовність",
                    min_value=0,
                    max_value=1,
                ),
                "liveness": st.column_config.ProgressColumn(
                    "Прямий етер",
                    min_value=0,
                    max_value=1,
                ),
                "tempo": st.column_config.NumberColumn(
                    "Темп",
                    format="%d",
                ),
                "energy": st.column_config.ProgressColumn(
                    "Енергійність",
                    min_value=0,
                    max_value=1,
                ),
                "instrumentalness": st.column_config.ProgressColumn(
                    "Інструментальність",
                    min_value=0,
                    max_value=1,
                ),
            },
            column_order=["track_name", "album_name", "artists", "artist_genres",
                          "danceability", "valence", "acousticness", "speechiness",
                          "liveness", "tempo", "energy", "instrumentalness"],
            hide_index=True,
            use_container_width=True
        )

        # Generate recommendations Spotify playlist embedding
        if not recommendations.empty:
            create_playlist_result = self.__spotify_client.create_playlist(recommendations['track_id'].values)
            if not create_playlist_result.is_success:
                st.error(create_playlist_result.message)
                return
            created_playlist_id = create_playlist_result.data['id']
            recommendation_playlist_embedding = f'''
            <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/{created_playlist_id}?utm_source=generator"
            width="100%" height="440" frameBorder="0" allowfullscreen=""
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
            '''
            st.markdown(recommendation_playlist_embedding, unsafe_allow_html=True)
        else:
            st.warning("Не знайдено треків для вибраного жанру та параметрів.")

    @staticmethod
    @st.cache_resource
    def get_data_model():
        data_model = KNNDataReaderPkl(
            "C:\\Users\\dbala\\Documents\\repos\\University\\Data\\content_based_kmeans_dataset.pkl"
        ).get_data_model()
        return data_model

# main
st.set_page_config(page_title="K-Nearest Neighbors", page_icon="🎷")
page = KNearestNeighborsRecommendationsStreamlitPage()
