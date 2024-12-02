import pandas as pd

class KNNRecommenderDataModel:
    def __init__(self, global_dataset: pd.DataFrame, audio_features: list):
        self.global_dataset = global_dataset
        self.audio_features = audio_features
