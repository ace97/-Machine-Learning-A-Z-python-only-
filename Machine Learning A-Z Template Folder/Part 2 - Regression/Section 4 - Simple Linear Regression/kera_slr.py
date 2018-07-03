from keras.layers import Dense, Activation
from keras.models import Sequential
import pandas as pd


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 0)

model = Sequential()
model.add(Dense(1,input_dim=1,bias_initializer='uniform',activation='relu'))
model.add(Activation('linear'))
model.compile(optimizer='sgd',loss='mse',metrics=['mse'])
model.fit(X_train,y_train, batch_size=1, epochs=30, shuffle=False)

ynew = model.predict(X_test)
# show the inputs and predicted outputs
for i in range(len(X_test)):
	print("X=%s, Predicted=%s Actual=%s" % (X_test[i], ynew[i],y_test[i]))

