import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, -1].values
Y = dataset.iloc[:, 3].values

# Split into training and testing sets if viable

'''

The DTR model is not a continuous (and non-linear) model.
For visualisation, high-resolution plotting must be used

DTR is not particularly useful for 1-D datasets.
but can provide interesting results in multi-dimensional models.
'''

# Fitting the DTR to dataset
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state = 0)
dt_regressor.fit(X, Y)

Y_pred = dt_regressor.predict(X)

X_grid = np.arange(min(X), max(X), 0.01) # 0.1 is the step size
X_grid = X_grid.reshape((len(X_grid), 1)) # number of lines, number of columns

# Plotting individual datapoints
plt.scatter(X, Y, color = 'red')
# Plotting predicted curve
plt.plot(X_grid, dt_regressor.predict(X_grid), color = 'blue')
plt.title('Title for plot')
plt.xlabel('X_label')
plt.xlabel('Y_label')
plt.show()
