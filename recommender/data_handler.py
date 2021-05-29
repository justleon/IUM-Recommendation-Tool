import pandas as pd
from sklearn.model_selection import train_test_split
import math
from utils import load_jsonl_pd

SEED = 1
MIN_INTERACTIONS = 5
TEST_RATIO = 0.2
VAL_RATIO = 0.2
TRAIN_RATIO = 1 - TEST_RATIO - VAL_RATIO
VIEW_PRODUCT_STRENGTH = 1
BUY_PRODUCT_STRENGTH = 3


def merge(sessions: pd.DataFrame, products: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(sessions, users, on='user_id')
    merged = pd.merge(merged, products, on='product_id')
    return merged


def smooth_preference(x: int):
    return math.log(1 + x, 2)


def get_items_interacted(user_id: int, data_set: pd.DataFrame) -> set[int]:
    interacted_items = data_set.loc[user_id]['product_id']
    return set(interacted_items)


class DataHandler:
    def __init__(self):
        sessions = load_jsonl_pd("data/sessions.jsonl")
        sessions = sessions.replace({'event_type': {'VIEW_PRODUCT': VIEW_PRODUCT_STRENGTH,
                                                    'BUY_PRODUCT': BUY_PRODUCT_STRENGTH}})
        self.products = load_jsonl_pd("data/products.jsonl")
        self.users = load_jsonl_pd("data/users.jsonl")

        interactions = sessions.merge(self.users, on='user_id')
        interactions = interactions.merge(self.products, on='product_id')

        interactions = interactions.groupby(['user_id', 'product_id'])['event_type'].sum() \
            .apply(smooth_preference).reset_index()

        interactions_train, interactions_test = train_test_split(interactions, stratify=interactions['user_id'],
                                                                 test_size=TEST_RATIO, random_state=SEED)

        interactions_train, interactions_val = train_test_split(interactions_train,
                                                                stratify=interactions_train['user_id'],
                                                                test_size=VAL_RATIO / (VAL_RATIO + TRAIN_RATIO),
                                                                random_state=SEED)

        # to speed up the search process during evaluation we index the sets
        self.interactions_indexed = interactions.set_index('user_id')
        self.interactions_train_indexed = interactions_train.set_index('user_id')
        self.interactions_test_indexed = interactions_test.set_index('user_id')
        self.interactions_val_indexed = interactions_val.set_index('user_id')
