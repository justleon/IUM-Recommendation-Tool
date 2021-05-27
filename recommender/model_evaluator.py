import random
from typing import Union

import pandas as pd
from data_handler import get_items_interacted, DataHandler
from models import PopularityBasedRecommender, ContentBasedRecommender

SEED = 1
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


def hit_top_n(product_id: int, recommended_items: list[int], top_n: int) -> int:
    return product_id in recommended_items[:top_n]


class ModelEvaluator:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def get_not_interacted_items_sample(self, user_id: int, sample_size: int) -> set[int]:
        interacted_items = get_items_interacted(user_id, self.data_handler.interactions_indexed)
        all_items = set(self.data_handler.products['product_id'])
        non_interacted = all_items - interacted_items
        random.seed(SEED)
        non_interacted_sample = random.sample(non_interacted, sample_size)
        return set(non_interacted_sample)

    def evaluate_model_for_user(self, model: Union[PopularityBasedRecommender, ContentBasedRecommender], user_id: int) \
            -> dict[str, Union[float]]:
        interacted_vals_test = self.data_handler.interactions_test_indexed.loc[user_id]
        person_interacted_items_test = set(interacted_vals_test['product_id'])
        interacted_items_count_test = len(person_interacted_items_test)

        user_recommendations = model.predict(user_id)
        hits_at_5 = 0
        hits_at_10 = 0

        for product_id in person_interacted_items_test:
            non_interacted_sample = self.get_not_interacted_items_sample(user_id,
                                                                         EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS)
            items_to_filter_recommendations = non_interacted_sample.union({product_id})
            valid_recommendations = \
                user_recommendations[user_recommendations['product_id'].isin(items_to_filter_recommendations)]
            hits_at_5 += int(hit_top_n(product_id, valid_recommendations['product_id'].tolist(), 5))
            hits_at_10 += int(hit_top_n(product_id, valid_recommendations['product_id'].tolist(), 10))

        rate_at_5 = hits_at_5 / float(interacted_items_count_test)
        rate_at_10 = hits_at_10 / float(interacted_items_count_test)

        user_metrics = {
            'hits@5_count': hits_at_5,
            'hits@10_count': hits_at_10,
            'interacted_count': interacted_items_count_test,
            'recall@5': rate_at_5,
            'recall@10': rate_at_10
        }
        return user_metrics

    def evaluate(self, model: Union[PopularityBasedRecommender, ContentBasedRecommender]) -> dict[str, float]:
        users_metrics = []
        for idx, user_id in enumerate(list(self.data_handler.interactions_test_indexed.index.unique().values)):
            user_metrics = self.evaluate_model_for_user(model, user_id)
            users_metrics.append(user_metrics)

        detailed_results = pd.DataFrame(users_metrics).sort_values('interacted_count', ascending=False)
        global_rate_at_5 = detailed_results['hits@5_count'].sum() / float(detailed_results['interacted_count'].sum())
        global_rate_at_10 = detailed_results['hits@10_count'].sum() / float(detailed_results['interacted_count'].sum())

        global_metrics = {
            'rate@5': global_rate_at_5,
            'rate@10': global_rate_at_10
        }
        return global_metrics
