import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utils import load_jsonl_pd, load_jsonl
from sklearn.metrics import adjusted_mutual_info_score
from itertools import combinations
from datetime import datetime


def mi_coefficient():
    u_cities = dict()
    u_names = dict()
    u_streets = dict()
    users = load_jsonl("data/users.jsonl")
    for user in users:
        u_cities[user['user_id']] = user['city']
        u_names[user['user_id']] = user['name']
        u_streets[user['user_id']] = user['street']

    p_prices = dict()
    p_categories = dict()
    p_names = dict()
    products = load_jsonl("data/products.jsonl")
    for product in products:
        p_categories[product['product_id']] = product['category_path']
        p_prices[product['product_id']] = product['price']
        p_names[product['product_id']] = product['product_name']

    s_user_ids = list()
    s_product_ids = list()
    s_event_types = list()
    s_offered_discounts = list()
    s_purchase_ids = list()
    sessions = load_jsonl("data/sessions.jsonl")
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

    dataframe = pd.DataFrame(d)

    dataframe['user_city'] = dataframe['user_id'].map(u_cities)
    dataframe['user_name'] = dataframe['user_id'].map(u_names)
    dataframe['user_street'] = dataframe['user_id'].map(u_streets)
    dataframe['product_name'] = dataframe['product_id'].map(p_names)
    dataframe['product_price'] = dataframe['product_id'].map(p_prices)
    dataframe['product_category'] = dataframe['product_id'].map(p_categories)

    comb = combinations(dataframe.columns.values, 2)

    for i in list(comb):
        print(i[0], '&', i[1], '->', round(adjusted_mutual_info_score(dataframe[i[0]], dataframe[i[1]]), 3))


def calc_percentage(part, whole, digits):
    percentage = 100 * float(part) / float(whole)
    rounded_percentage = round(percentage, digits)
    return str(rounded_percentage) + '%'


def make_percentage(number, digits):
    percentage = 100 * number
    rounded_percentage = round(percentage, digits)
    return str(rounded_percentage) + '%'


def prices_analysis():
    products = load_jsonl_pd("data/products.jsonl")

    ceny = []
    negativeVal = 0
    enormousVal = 0
    for i in products['price']:
        ceny.append(i)
    ceny.sort()
    numCeny = len(ceny)
    print("\nPrices Analysis:\n")
    print(products['price'].describe())
    for i in ceny:
        if float(i) <= 0:
            negativeVal += 1
        elif float(i) >= 10000:
            enormousVal += 1

    correctPrices = negativeVal + enormousVal
    print('All products: ', numCeny)
    print('Correct prices: ', numCeny - negativeVal - enormousVal)
    print('% of correct prices: ', (numCeny - correctPrices) / numCeny * 100)


def plot_data():
    products = load_jsonl_pd("data/products.jsonl")
    sessions = load_jsonl_pd("data/sessions.jsonl")
    merge = pd.merge(sessions, products, on='product_id', how='left')

    categories = set()
    for value in products["category_path"]:
        categories.update(value.split(";"))

    categories_count = {}
    for category in categories:
        categories_count[category] = 0

    for value in merge['category_path']:
        split = value.split(";")
        for category in split:
            categories_count[category] = categories_count.get(category) + 1
    print(categories_count)

    plt.figure(figsize=(5, 5))
    products['price'].plot.box()
    plt.gca().set_yscale("log")
    plt.savefig('../img/prices.png')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.hist(sessions['event_type'], facecolor='blue')
    plt.title('Distribution of event types')
    plt.savefig('../img/event_types.png')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.hist(sessions['user_id'], 200, facecolor='blue')
    plt.title('Occurrences of user ids')
    plt.xlabel('user id')
    plt.savefig('../img/user_ids.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.xticks(rotation='vertical')
    plt.bar(list(categories_count.keys()), list(categories_count.values()), facecolor='blue')
    plt.title('Occurrences of product categories')
    plt.tight_layout()
    plt.savefig('../img/product_categories.png')
    plt.show()


if __name__ == '__main__':
    valid_events, all_events, valid_or_repr_events = 0, 0, 0
    purchases, views = 0, 0
    user_ids = set()
    users_data = load_jsonl("data/users.jsonl")
    for user in users_data:
        user_ids.add(user['user_id'])

    product_ids = set()
    products_data = load_jsonl("data/products.jsonl")
    for product in products_data:
        product_ids.add(product['product_id'])

    matrix = np.zeros([len(user_ids), len(product_ids)])
    df = pd.DataFrame(matrix, columns=product_ids, index=user_ids)
    sessions_dict = dict()
    dates_set = set()
    events = list()
    events_data = load_jsonl("data/sessions.jsonl")
    for event in events_data:
        if event['user_id'] is not None:
            sessions_dict[event['session_id']] = event['user_id']
            if event['product_id'] is not None:
                valid_events += 1
        all_events += 1
    for event in events_data:
        if event['product_id'] is not None:
            if event['session_id'] in sessions_dict.keys():
                df[event['product_id']][sessions_dict[event['session_id']]] = 1
                valid_or_repr_events += 1
                dates_set.add(datetime.strptime(event['timestamp'], '%Y-%m-%dT%H:%M:%S').date())
                if event['event_type'] == 'BUY_PRODUCT':
                    purchases += 1
                elif event['event_type'] == 'VIEW_PRODUCT':
                    views += 1

    print('Users:', len(user_ids))
    print('Products:', len(product_ids))
    print('All events:', all_events)
    print('Initially valid events:', valid_events, calc_percentage(valid_events, all_events, 2))
    print('Valid or reproducible events:', valid_or_repr_events, calc_percentage(valid_or_repr_events, all_events, 2))
    print('Density', make_percentage(np.average(df.to_numpy()), 2))
    print('Daily sessions:', round(len(sessions_dict) / len(dates_set)))
    print('Daily views:', round(views / len(dates_set)))
    print('Daily purchases:', round(purchases / len(dates_set)))

    prices_analysis()
    plot_data()
    # mi_coefficient()
