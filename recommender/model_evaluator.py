import random
from typing import Union

import pandas as pd
from data_handler import get_interacted_products, DataHandler
from models import PopularityBasedRecommender, ContentBasedRecommender

SEED = 1
NON_INTERACTED_PRODUCTS_SAMPLE = 100


def hit_top_n(product_id: int, recommended_products: list[int], top_n: int) -> int:
    return product_id in recommended_products[:top_n]


class ModelEvaluator:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def get_not_interacted_products_sample(self, user_id: int, sample_size: int) -> set[int]:
        interacted_products = get_interacted_products(user_id, self.data_handler.interactions_indexed)
        all_products = set(self.data_handler.products['product_id'])
        non_interacted_products = all_products - interacted_products
        random.seed(SEED)
        non_interacted_products_sample = random.sample(non_interacted_products, sample_size)
        return set(non_interacted_products_sample)

    def evaluate_model_for_user(self, model: Union[PopularityBasedRecommender, ContentBasedRecommender], user_id: int,
                                interactions: pd.DataFrame) -> dict[str, Union[float]]:
        user_interactions = interactions.loc[user_id]
        interacted_products = set(user_interactions['product_id'])
        interacted_products_count = len(interacted_products)

        user_recommendations = model.predict(user_id)
        hits_at_5 = 0
        hits_at_10 = 0

        for product_id in interacted_products:
            non_interacted_sample = self.get_not_interacted_products_sample(user_id, NON_INTERACTED_PRODUCTS_SAMPLE)
            products_to_filter_recommendations = non_interacted_sample.union({product_id})
            valid_recommendations = \
                user_recommendations[user_recommendations['product_id'].isin(products_to_filter_recommendations)]
            hits_at_5 += int(hit_top_n(product_id, valid_recommendations['product_id'].tolist(), 5))
            hits_at_10 += int(hit_top_n(product_id, valid_recommendations['product_id'].tolist(), 10))

        rate_at_5 = hits_at_5 / float(interacted_products_count)
        rate_at_10 = hits_at_10 / float(interacted_products_count)

        user_metrics = {
            'hits@5_count': hits_at_5,
            'hits@10_count': hits_at_10,
            'interacted_count': interacted_products_count,
            'recall@5': rate_at_5,
            'recall@10': rate_at_10
        }
        return user_metrics

    def evaluate(self, model: Union[PopularityBasedRecommender, ContentBasedRecommender], interactions: pd.DataFrame) \
            -> dict[str, float]:
        users_metrics = []
        for idx, user_id in enumerate(list(interactions.index.unique().values)):
            user_metrics = self.evaluate_model_for_user(model, user_id, interactions)
            users_metrics.append(user_metrics)

        detailed_results = pd.DataFrame(users_metrics).sort_values('interacted_count', ascending=False)
        global_rate_at_5 = detailed_results['hits@5_count'].sum() / float(detailed_results['interacted_count'].sum())
        global_rate_at_10 = detailed_results['hits@10_count'].sum() / float(detailed_results['interacted_count'].sum())

        global_metrics = {
            'rate@5': global_rate_at_5,
            'rate@10': global_rate_at_10
        }
        return global_metrics
