import implicit

class MatrixFactorizationRecommendationEngine:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.model = None

    def train_model(self, factors=150, iterations=20, regularization=0.01):
        self.model = implicit.cpu.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization
        )
        self.model.fit(self.user_item_matrix.T)

    def recommend_artists(self, user_items, N=10):
        recommended, scores = self.model.recommend(
            userid=0,
            user_items=user_items,
            N=N
        )
        return recommended, scores
