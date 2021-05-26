import pandas as pd
from sklearn.model_selection import train_test_split
import math
from utils import load_jsonl_pd

MIN_INTERACTIONS = 5
TEST_SIZE = 0.2


def merge(
        sessions: pd.DataFrame, products: pd.DataFrame, users: pd.DataFrame
):
    merged = pd.merge(sessions, users, on='user_id')
    merged = pd.merge(merged, products, on='product_id')
    return merged

def smooth_preference(x):
    return math.log(1 + x, 2)

class DataHandler:
    def __init__(self):
        sessions = load_jsonl_pd("data/sessions.jsonl")
        sessions = sessions.replace({'event_type': {'VIEW_PRODUCT': 1, 'BUY_PRODUCT': 3}})
        self.products = load_jsonl_pd("data/products.jsonl")
        self.users = load_jsonl_pd("data/users.jsonl")
        self.sessions = merge(sessions, self.products, self.users)
        # self.sessions.drop('street', axis='columns', inplace=True)

        user_interactions_num = sessions.groupby(['user_id', 'product_id']).size().groupby('user_id').size()
        users_enough_interactions = user_interactions_num[user_interactions_num >= MIN_INTERACTIONS] \
            .reset_index(['user_id'])
        print("users: " + str(len(user_interactions_num)))
        interactions = sessions.merge(users_enough_interactions, how='right', left_on='user_id', right_on='user_id')
        self.interactions = interactions.groupby(['user_id', 'product_id'])['event_type'].sum() \
            .apply(smooth_preference).reset_index()
        print("total unique interactions: " + str(len(interactions)))
        interactions_train, interactions_test = train_test_split(interactions, stratify=interactions['user_id'],
                                                                 test_size=TEST_SIZE, random_state=1)
        print("train set: " + str(len(interactions_train)))
        print("test set: " + str(len(interactions_test)))

        self.interactions_indexed = interactions.set_index('user_id')
        self.interactions_train_indexed = interactions_train.set_index('user_id')
        self.interactions_test_indexed = interactions_test.set_index('user_id')
        self.item_popularity = self.interactions.groupby('product_id')['event_type'].sum() \
            .sort_values(ascending=False).reset_index()

    def get_items_interacted(self, user_id: int, data_set):
        interacted_items = data_set.loc[user_id]['product_id']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
