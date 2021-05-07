import pandas as pd
import matplotlib.pyplot as plt
import collections as col
import numpy as np

prices = pd.read_json("products.jsonl", lines=True)
#missing = pd.DataFrame({'total_missing': sessions.isnull().sum(), 'percent': (sessions.isnull().sum() / 84296) * 100})
#print(missing)
column = 'price'
ceny = []
negativeVal = 0
enormousVal = 0
for i in prices[column]:
    ceny.append(i)
ceny.sort()
numCeny = len(ceny)
print(prices[column].describe())
for i in ceny:
    print(i)
    if float(i) <= 0: negativeVal += 1
    elif float(i) >= 10000: enormousVal += 1

correctPrices = negativeVal + enormousVal
print('All products: ', numCeny)
print('Correct prices: ', numCeny - negativeVal - enormousVal)
print('% of correct prices: ', (numCeny - correctPrices) / numCeny * 100)

plt.figure(figsize=(5, 5))
prices[column].plot.box()
plt.gca().set_yscale("log")
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(prices[column][prices[column].notnull()], 500, facecolor='blue')
plt.gca().set_xscale("log")
plt.show()
