import streamlit as st
import pandas as pd
import scipy.sparse as sp

from pages.helpers.streamlit_ui_helper import StreamlitUiHelper
from src.services.content_based_recommendation.matrix_factorization_data_preprocessor import MatrixFactorizationDataPreprocessor
from src.services.content_based_recommendation.matrix_factorization_recommendation_engine import MatrixFactorizationRecommendationEngine

class MatrixFactorizationRecommendationsStreamlitPage:
    def __init__(self):
        self.init_content()
        self.init_sidebar()
        StreamlitUiHelper.hide_deploy_button()

    @staticmethod
    def init_content():
        st.markdown("# Рекомендації на основі матричної факторизації")
        st.markdown("#### Завантажте вашу історію прослуховувань у форматі JSON для отримання персоналізованих рекомендацій.")

        with st.expander("Детальніше про матричну факторизацію"):
            st.markdown("""
            Матрична факторизація - це техніка, яка використовується для створення рекомендацій на основі колаборативної фільтрації.
            Вона розкладає велику матрицю користувачів та предметів на добуток двох менших матриць, що представляють латентні фактори.
            """)

    def init_sidebar(self):
        st.sidebar.header("Завантаження історії прослуховувань")
        uploaded_file = st.sidebar.file_uploader("Завантажте файл JSON з історією прослуховувань", type="json")

        if uploaded_file is not None:
            with st.spinner("Обробка вашої історії прослуховувань..."):
                self.process_user_history(uploaded_file)

    def process_user_history(self, uploaded_file):
        my_listening_history = pd.read_json(uploaded_file)

        my_listening_history = my_listening_history.groupby(['artistName'], as_index=False).size().reset_index(name='count')

        data_preprocessor = MatrixFactorizationDataPreprocessor(
            user_artists_path='C:\\Users\\dbala\\Documents\\repos\\University\\Data\\collaborative_based_data\\lastfm_artists_dataset\\user_artists.dat',
            artists_path='C:\\Users\\dbala\\Documents\\repos\\University\\Data\\collaborative_based_data\\lastfm_artists_dataset\\artists.dat'
        )
        data_preprocessor.load_data()
        data_preprocessor.preprocess_data()

        my_listening_history_merged = pd.merge(
            my_listening_history,
            data_preprocessor.artists,
            left_on='artistName',
            right_on='name',
            how='inner'
        )

        if my_listening_history_merged.empty:
            st.error("Не вдалося знайти відповідності для ваших артистів у нашій базі даних.")
            return

        my_listening_history_merged['artist_index'] = my_listening_history_merged['id'].map(data_preprocessor.artist_id_to_index)
        my_listening_history_merged = my_listening_history_merged.dropna(subset=['artist_index'])
        my_listening_history_merged['artist_index'] = my_listening_history_merged['artist_index'].astype(int)

        if my_listening_history_merged.empty:
            st.error("Не вдалося знайти відповідності для ваших артистів у нашій базі даних.")
            return

        num_items = data_preprocessor.user_item_matrix.shape[1]

        user_items = sp.csr_matrix(
            (my_listening_history_merged['count'],
             ([0]*len(my_listening_history_merged), my_listening_history_merged['artist_index'].values)),
            shape=(1, num_items)
        )

        recommendation_engine = MatrixFactorizationRecommendationEngine(data_preprocessor.user_item_matrix)
        recommendation_engine.train_model()

        recommended_artist_indices, scores = recommendation_engine.recommend_artists(user_items)

        index_to_artist_id = data_preprocessor.index_to_artist_id
        recommended_artist_ids = [index_to_artist_id[idx] for idx in recommended_artist_indices]

        recommended_artists = [data_preprocessor.get_artist_name_by_id(artist_id) for artist_id in recommended_artist_ids]

        st.write('### Рекомендації артистів для вас:')
        for idx, artist_name in enumerate(recommended_artists, 1):
            st.write(f"{idx}. {artist_name}")

st.set_page_config(page_title="Matrix Factorization Recommendations", page_icon="🎧")
page = MatrixFactorizationRecommendationsStreamlitPage()
