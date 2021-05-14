import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mi_coefficient import mi_coefficient
from datetime import datetime


def load_jsonl_pd(path):
    return pd.read_json(path, lines=True)


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


def calc_percentage(part, whole, digits):
    percentage = 100 * float(part) / float(whole)
    rounded_percentage = round(percentage, digits)
    return str(rounded_percentage) + '%'


def make_percentage(number, digits):
    percentage = 100 * number
    rounded_percentage = round(percentage, digits)
    return str(rounded_percentage) + '%'


def prices_analysis():
    products = load_jsonl_pd("products.jsonl")

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
    products = load_jsonl_pd("products.jsonl")
    sessions = load_jsonl_pd("sessions.jsonl")
    merge = pd.merge(sessions, products, on='product_id', how='left')

    # categories = set()
    # for value in merge["category_path"]:
    #     categories.update(value.split(";"))
    # print(categories)

    plt.figure(figsize=(5, 5))
    products['price'].plot.box()
    plt.gca().set_yscale("log")
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.hist(sessions['event_type'], facecolor='blue')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.hist(sessions['user_id'], 200, facecolor='blue')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(merge['category_path'], 200, facecolor='blue')
    plt.show()


if __name__ == '__main__':
    valid_events, all_events, valid_or_repr_events = 0, 0, 0
    purchases, views = 0, 0
    user_ids = set()
    users_data = load_jsonl("users.jsonl")
    for user in users_data:
        user_ids.add(user['user_id'])

    product_ids = set()
    products_data = load_jsonl("products.jsonl")
    for product in products_data:
        product_ids.add(product['product_id'])

    matrix = np.zeros([len(user_ids), len(product_ids)])
    df = pd.DataFrame(matrix, columns=product_ids, index=user_ids)
    sessions_dict = dict()
    dates_set = set()
    events = list()
    events_data = load_jsonl("sessions.jsonl")
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
