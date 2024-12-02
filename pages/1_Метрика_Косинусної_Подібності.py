from ast import literal_eval
from pathlib import Path

import streamlit as st
from plotly.subplots import make_subplots

from pages.helpers.streamlit_ui_helper import StreamlitUiHelper
from src import constants
from src.clients.spotify_client import SpotifyClient
from src.services.content_based_recommendation.cosine_similarity_recommendation_engine import CosineSimilarityRecommendationEngine
from src.services.data_readers.cosine_similarity_data_reader import CosineSimilarityDataReaderPkl
import numpy as np
import plotly.graph_objects as go
import pandas as pd

class CosineSimilarityRecommendationsStreamlitPage:
    def __init__(self):
        self.playlist_raw_link = ''
        self.playlist_size = 0
        self.__spotify_client = SpotifyClient("02aca3b049a74aeba6a820a0ba7e0c50", "513bcf2584eb43518a808ad653dfa9b8")
        self.init_content()
        self.init_sidebar()
        StreamlitUiHelper.hide_deploy_button()

    @staticmethod
    def init_content():
        st.markdown("# РекомендаціЇ з використанням метрики косинусної подібності")
        with st.expander("Детальніше про косинус подібності"):
            path = Path(__file__).parent / "../src/markdowns/cosine_similarity_detailed_description.md"
            print(path)
            st.markdown(open(path, encoding="utf8").read())

    def init_sidebar(self):
        placeholder = st.empty()
        st.sidebar.header("Параметри плейлиста")

        with st.sidebar.form(key='Form1'):
            self.playlist_raw_link: str = st.text_input('Введіть посилання на вхідний плейлист')
            self.playlist_size: int = st.slider('Розмір рекомендаційного плейлиста', 5, 20, 12, 1)
            submitted = st.form_submit_button(label='Знайти рекомендації 🔎')

        if submitted:
            placeholder.empty()
            with placeholder.container():
                with st.spinner("Генерація рекомендацій"):
                    self.generate_recommendations()

    def generate_recommendations(self):
        if self.playlist_raw_link == '' or self.playlist_size == 0:
            st.sidebar.warning('Введіть посилання на плейлист. Він має бути публічним, для того, щоб можна було зчитати'
                               'аудіо характеристики.')
            return

        playlist_uri = self.__spotify_client.get_playlist_uri_by_raw_link(self.playlist_raw_link)

        # дані пісень плейлиста
        playlist_info = self.__spotify_client.get_playlist_info_by_playlist_uri(playlist_uri)
        playlist_tracks = self.__spotify_client.get_spotify_tracks_by_playlist_uri(playlist_uri)
        if not playlist_tracks.is_success:
            st.error(playlist_tracks.message)
            return
        st.toast('Опрацювання рекомендацій', icon='⚙️')

        playlist_name = playlist_info.data['name']
        st.write('# Аналіз плейлиста: ' + playlist_name)

        global_features_model = self.get_global_features_data()
        print(global_features_model)
        dataset_numeric_sparse = global_features_model.dataset_numeric_sparse
        dataset_without_genres = global_features_model.dataset_without_genres
        columns_map = global_features_model.columns_map

        track_ids = [track['track']['id'] for track in playlist_tracks.data['items']]
        client_feature_indices = dataset_without_genres[dataset_without_genres['track_id'].isin(track_ids)]['track_id'].index.to_numpy()
        client_playlist_feature_vector = np.asarray(dataset_numeric_sparse[client_feature_indices].mean(axis=0))
        client_df = dataset_without_genres[dataset_without_genres['track_id'].isin(track_ids)]

        self.show_mean_audio_features_analysis(client_df)
        self.show_features_distibution(client_df)
        self.show_genre_breakdown(client_df)
        self.show_popularity(client_df)

        # Ініціалізація CosineSimilarityRecommendationEngine з глобальним датасетом
        generator = CosineSimilarityRecommendationEngine(dataset_numeric_sparse, dataset_without_genres)

        # Генеруй рекомендації, виключаючи треки в клієнтському плейлисті
        recommendations = generator.generate_recommendations(client_playlist_feature_vector, client_feature_indices, self.playlist_size, False)

        # Випадайка з описом властиовстей
        with st.expander("Опис якостей"):
            path = Path(__file__).parent / "../src/markdowns/features_description.md"
            print(path)
            st.markdown(open(path, encoding="utf8").read())

        # Рекомендації пісень
        st.write('### Рекомендації пісень')
        recommendations_df = dataset_without_genres[
            dataset_without_genres['track_id'].isin(recommendations[:, 6])]
        st.dataframe(
            recommendations_df[
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

        # Згенеруй програвач пісень Spotify
        create_playlist_result = self.__spotify_client.create_playlist(recommendations_df['track_id'].values)
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

    def show_mean_audio_features_analysis(self, client_df):
        st.write('### Середнє по аудіо властивостях')
        col1, col2, col3 = st.columns(3)
        # TODO: Better parse it from spotify instead of local dataset
        client_playlist_feature_vector = (client_df[['energy', 'danceability', 'valence', 'tempo', 'loudness',
                                                     'acousticness', 'instrumentalness', 'liveness', 'speechiness']]
                                          .mean(axis=0))
        counter = 0
        for index, value in client_playlist_feature_vector.items():
            if counter % 3 == 0:
                col1.metric(constants.FEATURES_ANALYSIS_MAP[index], f"{value:.2f}")
            if counter % 3 == 1:
                col2.metric(constants.FEATURES_ANALYSIS_MAP[index], f"{value:.2f}")
            if counter % 3 == 2:
                col3.metric(constants.FEATURES_ANALYSIS_MAP[index], f"{value:.2f}")
            counter += 1

    def show_genre_breakdown(self, client_df):
        st.markdown("### Секторна діаграма жанрів")

        genres_df: pd.DataFrame = client_df['artist_genres'].apply(literal_eval).to_frame().explode("artist_genres")
        trace = go.Pie(labels=genres_df['artist_genres'], values=genres_df['artist_genres'].value_counts().values,
                       hole=.3)
        genre_pie_chart = go.Figure(data=[trace])
        genre_pie_chart.update_layout(transition_duration=500)
        st.plotly_chart(genre_pie_chart)

    def show_popularity(self, client_df):
        # Створення гістограми
        fig = make_subplots(rows=1, cols=2, shared_yaxes=False)

        fig.add_trace(go.Bar(x=[x[:30] for x in client_df['track_name']], y=client_df["track_popularity"],
                             name="Популярність треків"), 1, 1)
        fig.add_trace(go.Bar(x=[x[:30] for x in client_df['album_name']], y=client_df["album_popularity"],
                             name="Популярність альбомів"), 1, 2)

        fig.update_xaxes(tickangle=45)
        fig.update_layout(title=dict(
            text='Аналіз популярності',
            font=dict(size=25)
        ))
        st.plotly_chart(fig)

    def show_features_distibution(self, client_df):
        fig = make_subplots(rows=3, cols=3, shared_yaxes=True, subplot_titles=(
            '<i>Тривалість,мс', '<i>Танцювальність', '<i>Енергійність', '<i>Гучність', '<i>Мовність', '<i>Акустичність',
            '<i>Прямий етер', '<i>Позитивність', '<i>Темп'))
        fig.add_trace(go.Histogram(x=client_df['duration_ms'], name='Тривалість,мс'), row=1, col=1)
        fig.add_trace(go.Histogram(x=client_df['danceability'], name='Танцювальність'), row=1, col=2)
        fig.add_trace(go.Histogram(x=client_df['energy'], name='Енергійність'), row=1, col=3)
        fig.add_trace(go.Histogram(x=client_df['loudness'], name='Гучність'), row=2, col=1)
        fig.add_trace(go.Histogram(x=client_df['speechiness'], name='Мовність'), row=2, col=2)
        fig.add_trace(go.Histogram(x=client_df['acousticness'], name='Акустичність'), row=2, col=3)
        fig.add_trace(go.Histogram(x=client_df['liveness'], name='Прямий етер'), row=3, col=1)
        fig.add_trace(go.Histogram(x=client_df['valence'], name='Позитивність'), row=3, col=2)
        fig.add_trace(go.Histogram(x=client_df['tempo'], name='Темп'), row=3, col=3)
        fig.update_layout(height=900, width=900, title=dict(
            text='Гістограми аудіо якостей вхідного плейлиста',
            font=dict(size=25)
        ))
        fig.update_layout(template='plotly_dark', bargap=0.01)
        fig.update_yaxes(title="Кількість треків")

        st.plotly_chart(fig)

    @staticmethod
    @st.cache_data
    def get_global_features_data():
        return (CosineSimilarityDataReaderPkl(
            "C:\\Users\dbala\Documents\\repos\\University\\Data\\content_based_374349_songs_complete.pkl")
                .get_global_features_and_metadata())


st.set_page_config(page_title="Cosine Similarity", page_icon="🎷")
page = CosineSimilarityRecommendationsStreamlitPage()
