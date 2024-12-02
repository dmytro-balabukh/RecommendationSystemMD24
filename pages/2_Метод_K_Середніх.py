import streamlit as st
from sklearn.preprocessing import StandardScaler

from pages.helpers.streamlit_ui_helper import StreamlitUiHelper
from src import constants
from src.clients.spotify_client import SpotifyClient
from src.services.content_based_recommendation.kmeans_data_preprocessor import KMeansDataPreprocessor
from src.services.content_based_recommendation.kmeans_recommendation_engine import KMeansRecommendationEngine
from src.services.data_readers.k_means_data_reader import KMeansDataReaderPkl
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from ast import literal_eval
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

class KMeansClusteringRecommendationsStreamlitPage:
    def __init__(self):
        self.playlist_raw_link = ''
        self.playlist_size = 0
        self.__spotify_client = SpotifyClient("02aca3b049a74aeba6a820a0ba7e0c50", "513bcf2584eb43518a808ad653dfa9b8")
        self.init_content()
        self.init_sidebar()
        StreamlitUiHelper.hide_deploy_button()

    @staticmethod
    def init_content():
        st.markdown("# РекомендаціЇ методом k-середніх")

        with st.expander("Детальніше про k-середніх"):
            path = Path(__file__).parent / "../src/markdowns/kmeans_detailed_description.md"
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

        # дані пісень плейлиста із спотіфаю
        playlist_info = self.__spotify_client.get_playlist_info_by_playlist_uri(playlist_uri)
        playlist_tracks = self.__spotify_client.get_spotify_tracks_by_playlist_uri(playlist_uri)
        if not playlist_tracks.is_success:
            st.error(playlist_tracks.message)
            return
        st.toast('Опрацювання рекомендацій', icon='⚙️')

        playlist_name = playlist_info.data['name']
        st.write('# Аналіз плейлиста: ' + playlist_name)

        data_model, scaler = self.get_data_and_scaler()

        track_ids = [track['track']['id'] for track in playlist_tracks.data['items']]
        print(track_ids)
        client_df = data_model.global_dataset[data_model.global_dataset['track_id'].isin(track_ids)]

        self.show_mean_audio_features_analysis(client_df)
        self.show_features_distibution(client_df)
        self.show_genre_breakdown(client_df)
        self.show_popularity(client_df)

        generator = KMeansRecommendationEngine()
        kmeans_model = self.get_kmeans_model(data_model.model_data_scaled)

        recommendations = generator.generate_recommendations(
            client_df,
            data_model.global_dataset,
            data_model.model_data_scaled,
            scaler,
            kmeans_model,
            self.playlist_size)

        with st.expander("Опис якостей"):
            path = Path(__file__).parent / "../src/markdowns/features_description.md"
            print(path)
            st.markdown(open(path, encoding="utf8").read())

        st.write('### Рекомендації пісень')
        recommendations_df = data_model.global_dataset[
            data_model.global_dataset['track_id'].isin(recommendations['track_id'])]
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

        if not recommendations_df.empty:
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
        else:
            st.warning("Не знайдено треків для вибраних параметрів.")

    def show_genre_breakdown(self, client_df):
        st.markdown("### Секторна діаграма жанрів")

        genres_df: pd.DataFrame = client_df['artist_genres'].apply(literal_eval).to_frame().explode("artist_genres")
        trace = go.Pie(labels=genres_df['artist_genres'], values=genres_df['artist_genres'].value_counts().values,
                       hole=.3)
        genre_pie_chart = go.Figure(data=[trace])
        genre_pie_chart.update_layout(transition_duration=500)
        st.plotly_chart(genre_pie_chart)


    def show_mean_audio_features_analysis(self, client_df):
        st.write('### Середнє по аудіо властивостях')
        col1, col2, col3 = st.columns(3)
        # TODO: Краще стягуй властивості із спотфіаю, ніж із глобального датасету
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

    def show_popularity(self, client_df):
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
        # fig.update_traces(marker_color='blue')
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
    def get_data_and_scaler():
        data_model = KMeansDataReaderPkl(
            # TODO: Add path to specific dataset for clustering
            "C:\\Users\\dbala\\Documents\\repos\\University\\Data\\content_based_kmeans_dataset.pkl").get_data_model()

        scaler = StandardScaler()
        KMeansDataPreprocessor(data_model).preprocess_model_dataset(scaler)

        return data_model, scaler

    @staticmethod
    @st.cache_resource
    def get_kmeans_model(model_data_scaled):
        model: KMeans = KMeans(
            n_clusters=11, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0,
        )
        model.fit(model_data_scaled)

        return model


st.set_page_config(page_title="K-Means Clustering", page_icon="🎷")
page = KMeansClusteringRecommendationsStreamlitPage()
