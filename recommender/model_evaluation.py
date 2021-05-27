import random
from typing import Union, Tuple

import pandas as pd
from data_handler import get_items_interacted, DataHandler
from models import PopularityBasedRecommender, ContentBasedRecommender

SEED = 1
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


class ModelEvaluator:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def get_not_interacted_items_sample(self, user_id: int, sample_size: int, seed: int = SEED) -> set[int]:
        random.seed(seed)

        interacted_items = get_items_interacted(user_id, self.data_handler.interactions_indexed)
        all_items = set(self.data_handler.products['product_id'])
        non_interacted = all_items - interacted_items

        non_interacted_sample = random.sample(non_interacted, sample_size)
        return set(non_interacted_sample)

    def if_hit_top_n(self, product_id: int, recommended_items: list[int], top_n: int) -> Tuple[int, int]:
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == product_id)
        except:
            index = -1
        hit = int(index in range(0, top_n))
        return hit, index

    def evaluate_model_for_user(self, model: Union[PopularityBasedRecommender, ContentBasedRecommender], user_id: int) \
            -> dict[str, Union[float]]:
        interacted_vals_test = self.data_handler.interactions_test_indexed.loc[user_id]
        if type(interacted_vals_test['product_id']) == pd.Series:
            person_interacted_items_test = set(interacted_vals_test['product_id'])
        else:
            person_interacted_items_test = {int(interacted_vals_test['product_id'])}
        interacted_items_count_test = len(person_interacted_items_test)

        user_recommendations = model.predict(user_id)
        hits_at_5_num = 0
        hits_at_10_num = 0

        for product_id in person_interacted_items_test:
            non_interacted_sample = self.get_not_interacted_items_sample(user_id,
                                                                         EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                         product_id % (2 ** 32))
            items_to_filter_recommendations = non_interacted_sample.union({product_id})
            valid_recommendations_temp = user_recommendations[
                user_recommendations['product_id'].isin(items_to_filter_recommendations)]
            valid_recommendations = valid_recommendations_temp['product_id'].values
            hit_at_5, index_at_5 = self.if_hit_top_n(product_id, valid_recommendations, 5)
            hits_at_5_num += hit_at_5
            hit_at_10, index_at_10 = self.if_hit_top_n(product_id, valid_recommendations, 10)
            hits_at_10_num += hit_at_10

        rate_at_5 = hits_at_5_num / float(interacted_items_count_test)
        rate_at_10 = hits_at_10_num / float(interacted_items_count_test)

        user_metrics = {
            'hits@5_count': hits_at_5_num,
            'hits@10_count': hits_at_10_num,
            'interacted_count': interacted_items_count_test,
            'recall@5': rate_at_5,
            'recall@10': rate_at_10
        }
        return user_metrics

    def evaluate(self, model: Union[PopularityBasedRecommender, ContentBasedRecommender]) -> dict[str, float]:
        users_metrics = []
        for idx, user_id in enumerate(list(self.data_handler.interactions_test_indexed.index.unique().values)):
            user_metrics = self.evaluate_model_for_user(model, user_id)
            user_metrics['_user_id'] = user_id
            users_metrics.append(user_metrics)
        # print('processed ' + str(idx) + ' users')

        detailed_results = pd.DataFrame(users_metrics).sort_values('interacted_count', ascending=False)
        global_rate_at_5 = detailed_results['hits@5_count'].sum() / float(detailed_results['interacted_count'].sum())
        global_rate_at_10 = detailed_results['hits@10_count'].sum() / float(detailed_results['interacted_count'].sum())

        global_metrics = {
            'rate@5': global_rate_at_5,
            'rate@10': global_rate_at_10
        }
        return global_metrics
