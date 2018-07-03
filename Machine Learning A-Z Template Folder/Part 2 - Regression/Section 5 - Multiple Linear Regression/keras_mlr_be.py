from keras.layers import Dense, Activation
from keras.models import Sequential
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

X = np.append(np.ones((50,1)).astype(float), values=X, axis=1)

def backwardElimination(x,sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size=.2, random_state=0)

model = Sequential()
model.add(Dense(units=1, bias_initializer="uniform", input_dim=2))
model.add(Activation('linear'))
model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.fit(X_train,y_train, batch_size=1, epochs=80, shuffle=False)

ynew = model.predict(X_test)

for i in range(len(X_test)):
	print("obs=%s, Predicted=%s Actual=%s" % (i, ynew[i],y_test[i]))
