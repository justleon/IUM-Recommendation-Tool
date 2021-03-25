import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data
    

if __name__ == '__main__':
    u_cities = list()
    users = load_jsonl("users.jsonl")
    for user in users:
        u_cities.append(user['city'])
    print("Users - cities: ")
    print('\t', "all: ", len(u_cities), ", unique: ", len(set(u_cities)), ", missing: ", u_cities.count(None))
    
    p_prices = list()
    p_categories = list()
    products = load_jsonl("products.jsonl")
    for product in products:
        p_categories.append(product['category_path'])
        p_prices.append(product['price'])
    print("Products - categories: ")
    print('\t', "all: ", len(p_categories), ", unique: ", len(set(p_categories)), ", missing: ", p_categories.count(None))
    print("Products - prices: ")
    print('\t', "all: ", len(p_prices), ", unique: ", len(set(p_prices)), ", missing: ", p_prices.count(None), ", min: ", min(p_prices), ", max: ", max(p_prices))

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
        s_purchase_ids.append(session['purchase_id'])
    print("Sessions - user ids: ")
    print('\t', "all: ", len(s_user_ids), ", unique: ", len(set(s_user_ids)), ", missing: ", s_user_ids.count(None))
    print("Sessions - product ids: ")
    print('\t', "all: ", len(s_product_ids), ", unique: ", len(set(s_product_ids)), ", missing: ", s_product_ids.count(None))
    print("Sessions - event types: ")
    print('\t', "all: ", len(s_event_types), ", unique: ", len(set(s_event_types)), ", missing: ", s_event_types.count(None))
    print("Sessions - offered discounts: ")
    print('\t', "all: ", len(s_offered_discounts), ", unique: ", len(set(s_offered_discounts)), ", missing: ", s_offered_discounts.count(None), ", min: ", min(s_offered_discounts), ", max: ", max(s_offered_discounts))
    print("Sessions - purchase ids: ")
    print('\t', "all: ", len(s_purchase_ids), ", unique: ", len(set(s_purchase_ids)), ", missing: ", s_purchase_ids.count(None))
