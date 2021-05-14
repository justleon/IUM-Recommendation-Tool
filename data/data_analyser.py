import pandas as pd
import matplotlib.pyplot as plt
import collections as col
import numpy as np

products = pd.read_json("products.jsonl", lines=True)
sessions = pd.read_json("sessions.jsonl", lines=True)
users = pd.read_json("users.jsonl", lines=True)

def prices_analysis():
    ceny = []
    negativeVal = 0
    enormousVal = 0
    for i in products['price']:
        ceny.append(i)
    ceny.sort()
    numCeny = len(ceny)
    print(products['price'].describe())
    for i in ceny:
        print(i)
        if float(i) <= 0: negativeVal += 1
        elif float(i) >= 10000: enormousVal += 1

    correctPrices = negativeVal + enormousVal
    print('All products: ', numCeny)
    print('Correct prices: ', numCeny - negativeVal - enormousVal)
    print('% of correct prices: ', (numCeny - correctPrices) / numCeny * 100)


if __name__ == '__main__':
    merge = pd.merge(sessions, products, on='product_id', how='left')
    categories = set()
    for value in merge["category_path"]:
        categories.update(value.split(";"))
    print(categories)

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
    plt.hist(merge['category_path'], 200, facecolor='blue', orientation='horizontal')
    plt.show()
