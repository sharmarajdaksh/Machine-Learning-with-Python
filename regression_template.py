import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, -1].values
Y = dataset.iloc[:, 3].values

# Splitting the dataset into Training and Tests sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

'''

# Fitting Linear Regression Model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Greater degree means greater accuracy and flexibility
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, Y_train)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Fitting Support Vector Regression (SVR)
# For SVR modelling, Feature Scaling is CRITICALLY IMPORTANT
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf') # most commonly used
svr_regressor.fit(X_train, Y_train)
# See SVR tutorial to see how to do the reverse process of Feature Scaling to get the originl value(s)


# Predicting Y-values for the test case
#Y_pred_linear = lin_reg.predict(X_test)
#Y_pred_poly = lin_reg_poly.predict(poly_reg.fit_transform(X_test))
#Y_pred = svr_regressor.predict(X_test)

#Plotting
# Works for single X value against single Y

# Plotting individual datapoints
plt.scatter(X, Y, color = 'red')
# Plotting predicted curve
plt.plot(X, Y_pred, color = 'blue')
plt.title('Title for plot')
plt.xlabel('X_label')
plt.xlabel('Y_label')
plt.show()

#Plotting (for higher resolution, smoother curve)
# Works for single X value against single Y

# A vector grid 
X_grid = np.arange(min(X), max(X), 0.1) # 0.1 is the step size
X_grid = X_grid.reshape((len(X_grid), 1)) # number of lines, number of columns

# Plotting individual datapoints
plt.scatter(X, Y, color = 'red')
# Plotting predicted curve
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Title for plot')
plt.xlabel('X_label')
plt.xlabel('Y_label')
plt.show()
