import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('machine.csv', header=None)   #.sample(frac=1)
X = dataset.iloc[:,2:-2].values
Y = dataset.iloc[:, 8].values

from sklearn.preprocessing import StandardScaler
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y.reshape(-1,1))
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

'''
NOT NEEDED
#Encode the first two columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:,1] = labelencoder.fit_transform(X[:,1]) 
onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder.fit_transform(X).toarray()

'''

#import statsmodels.formula.api as sm
#Add the B0 constant to our linear regression equation
#X = np.append(arr = np.ones((209, 1)).astype(int), values = X, axis = 1)

#Optimal Matrix of Features, instead of ALL the independent variables
#Only the most impactful variables are used
#X_opt = X[:, [0, 1, 2, 3, 6, 7]]

#New regressor
#regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#regressor_OLS.summary()

# Splitting the dataset into Training and Tests sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Fitting the RFR to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0) #n_estimators is the number of decision trees to be used
regressor.fit(X_train, Y_train)

'''
# Fitting Support Vector Regression (SVR)
# For SVR modelling, Feature Scaling is CRITICALLY IMPORTANT
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear') # most commonly used
regressor.fit(X_train, Y_train)
'''

'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
'''

'''
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
regressor = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=3, verbose=1)
regressor.fit(X_train, Y_train)
'''

Y_pred = regressor.predict(X_test)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(regressor, X_train, Y_train, cv=kfold)
print(results)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

history = regressor.fit(X_test, Y_test)
predicted = regressor.predict(X_test)
#accuracy_score(Y_test, prediction)
rmse = np.sqrt(((predicted - Y_test) ** 2).mean(axis=0))
print("Root Mean Squared Error: ",rmse)
print("Root Mean Squared Error: ", np.mean(rmse))


'''
from sklearn.metrics import mean_squared_error
loss_train = mean_squared_error(Y_train, regressor.predict(X_train))
loss_test = mean_squared_error(Y_pred, Y_test)
print("mse on train set: ", loss_train)
print("mse on test set: ", loss_test)
print("train score", regressor.score(X_train, Y_train))
print("test score", regressor.score(X_test, Y_test))
''' 