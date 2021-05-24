import pandas as pd
from utils import load_jsonl_pd


def merge(
        sessions: pd.DataFrame, products: pd.DataFrame, users: pd.DataFrame
):
    merged = pd.merge(sessions, users, on='user_id')
    merged = pd.merge(merged, products, on='product_id')
    return merged


class DataHandler:
    def __init__(self):
        sessions = load_jsonl_pd("data/sessions.jsonl")
        sessions = sessions.replace({'event_type': {'VIEW_PRODUCT': 1, 'BUY_PRODUCT': 3}})
        self.products = load_jsonl_pd("data/products.jsonl")
        self.users = load_jsonl_pd("data/users.jsonl")
        self.sessions = merge(sessions, self.products, self.users)
        # self.sessions.drop('street', axis='columns', inplace=True)
