import json
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from itertools import combinations


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


if __name__ == '__main__':
    u_cities = dict()
    u_names = dict()
    u_streets = dict()
    users = load_jsonl("users.jsonl")
    for user in users:
        u_cities[user['user_id']] = user['city']
        u_names[user['user_id']] = user['name']
        u_streets[user['user_id']] = user['street']

    p_prices = dict()
    p_categories = dict()
    p_names = dict()
    products = load_jsonl("products.jsonl")
    for product in products:
        p_categories[product['product_id']] = product['category_path']
        p_prices[product['product_id']] = product['price']
        p_names[product['product_id']] = product['product_name']

    s_user_ids = list()
    s_product_ids = list()
    s_event_types = list()
    s_offered_discounts = list()
    s_purchase_ids = list()
    sessions = load_jsonl("sessions.jsonl")
    for session in sessions:
        s_user_ids.append(session['user_id'])
        s_product_ids.append(session['product_id'])
        s_event_types.append(session['event_type'])
        s_offered_discounts.append(session['offered_discount'])
        if session['purchase_id'] is None:
            s_purchase_ids.append(0)
        else:
            s_purchase_ids.append(session['purchase_id'])

    d = {'user_id': s_user_ids,
         'product_id': s_product_ids,
         'event_type': s_event_types,
         'offered_discount': s_offered_discounts,
         'purchase_id': s_purchase_ids}

    df = pd.DataFrame(d)

    df['user_city'] = df['user_id'].map(u_cities)
    df['user_name'] = df['user_id'].map(u_names)
    df['user_street'] = df['user_id'].map(u_streets)
    df['product_name'] = df['product_id'].map(p_names)
    df['product_price'] = df['product_id'].map(p_prices)
    df['product_category'] = df['product_id'].map(p_categories)

    comb = combinations(df.columns.values, 2)

    for i in list(comb):
        print(i[0], '&', i[1], '->', round(adjusted_mutual_info_score(df[i[0]], df[i[1]]), 3))
