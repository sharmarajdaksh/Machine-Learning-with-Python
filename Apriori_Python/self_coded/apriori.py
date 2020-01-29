# Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
dataset = df.values

#convert each row to a list and make a list of all those lists
transactions = [ [ str(dataset[i, j])
                 for j in range(0, 20) ] for i in range(0, 7501) ]

#training apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, maximum_length = 2)

#visualise results
results = list(rules) #rules are sorted by default
