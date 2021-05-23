class PopularityBasedRecommender:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.product_popularity = self.data_handler.data.groupby('product_id').size().sort_values(ascending=False) \
            .reset_index(name='popularity')

    def predict(self, user_id: int):
        if user_id in set(self.data_handler.users['user_id']):
            return self.product_popularity['product_id'].to_list()
        return None
