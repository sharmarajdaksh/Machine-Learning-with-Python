import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, -1].values
Y = dataset.iloc[:, 3].values

# Split into training and testing sets if viable

'''

The RFR model is not a continuous (and non-linear) model.
For visualisation, high-resolution plotting must be used

The model essentially uses the average of a large number of decision trees
taken across our dataset, by choosing varying k data points for generating the trees.

The number of 'steps' or 'splits' increases, implying, possibly, a prediction closer to the actual value

'''

# Fitting the RFR to dataset
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 300, random_state = 0) #n_estimators is the number of decision trees to be used
rf_regressor.fit(X, Y)

Y_pred = rf_regressor.predict(X)

X_grid = np.arange(min(X), max(X), 0.01) # 0.1 is the step size
X_grid = X_grid.reshape((len(X_grid), 1)) # number of lines, number of columns

# Plotting individual datapoints
plt.scatter(X, Y, color = 'red')
# Plotting predicted curve
plt.plot(X_grid, rf_regressor.predict(X_grid), color = 'blue')
plt.title('Title for plot')
plt.xlabel('X_label')
plt.xlabel('Y_label')
plt.show()
