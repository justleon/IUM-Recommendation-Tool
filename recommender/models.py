import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

from data_handler import DataHandler
import pandas as pd
import numpy as np


class PopularityBasedRecommender:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler
        self.product_popularity = self.data_handler.sessions.groupby('product_id')['event_type'].size() \
            .sort_values(ascending=False).reset_index(name='popularity')
        self.product_popularity['rec_strength'] = \
            self.product_popularity['popularity'] / self.product_popularity['popularity'].max()

    def predict(self, user_id: int):
        if user_id in set(self.data_handler.users['user_id']):
            return self.product_popularity[['product_id', 'rec_strength']]
        return None


class ContentBasedRecommender:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler
        tf = TfidfVectorizer(use_idf=True)
        self.product_ids = self.data_handler.products['product_id'].tolist()
        self.tfidf_matrix = tf.fit_transform(self.data_handler.products['product_name'] + ';'
                                             + self.data_handler.products['category_path'])
        self.user_profiles = self.create_users_profiles()

    def create_product_profile(self, product_id: int):
        return self.tfidf_matrix[self.product_ids.index(product_id)]

    def create_product_profiles(self, ids: List[int]):
        return scipy.sparse.vstack([self.create_product_profile(x) for x in ids])

    def create_user_profile(self, user_id: int, indexed_sessions: pd.DataFrame):
        indexed_user_sessions = indexed_sessions.loc[user_id]
        user_product_profiles = self.create_product_profiles(indexed_user_sessions['product_id'])
        user_product_events = np.array(indexed_user_sessions['event_type']).reshape(-1, 1)
        return np.sum(user_product_profiles.multiply(user_product_events), axis=0) / np.sum(
            user_product_events)

    def create_users_profiles(self):
        indexed_sessions = self.data_handler.sessions.set_index('user_id')
        user_profiles = {}
        for user_id in indexed_sessions.index.unique():
            user_profiles[user_id] = self.create_user_profile(user_id, indexed_sessions)
        return user_profiles

    def predict(self, user_id: int):
        if user_id in set(self.data_handler.users['user_id']):
            cosine_similarities = cosine_similarity(self.user_profiles[user_id], self.tfidf_matrix)
            similar_indices = cosine_similarities.argsort().flatten()
            similar_items = sorted([(self.product_ids[i], cosine_similarities[0, i]) for i in similar_indices],
                                   key=lambda x: -x[1])
            return pd.DataFrame(similar_items, columns=['product_id', 'rec_strength'])
        return None