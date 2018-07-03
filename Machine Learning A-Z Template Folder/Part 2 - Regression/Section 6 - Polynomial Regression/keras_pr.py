from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

model = Sequential()
model.add(Dense(units=200, bias_initializer="uniform", input_dim=5, activation='relu'))
model.add(Dense(units=45))
model.add(Activation('relu'))
model.add(Dense(units=1))
model.add(Activation('linear'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
model.fit(X_poly, y, batch_size=1,epochs=15400)

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.05)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, model.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression with higher resolution)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print "salary predicted by polynomial model=",model.predict(poly_reg.fit_transform(6.5))