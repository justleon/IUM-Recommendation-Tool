import pandas as pd
import matplotlib.pyplot as plt
import collections as col
import numpy as np

products = pd.read_json("products.jsonl", lines=True)
sessions = pd.read_json("sessions.jsonl", lines=True)
column = 'price'
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

plt.figure(figsize=(5, 5))
products['price'].plot.box()
plt.gca().set_yscale("log")
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(products['price'], 500, facecolor='blue')
plt.gca().set_xscale("log")
plt.show()

plt.figure(figsize=(5, 5))
plt.hist(sessions['event_type'], facecolor='blue')
plt.show()

plt.figure(figsize=(5, 5))
plt.hist(sessions['user_id'], 200, facecolor='blue')
plt.show()
