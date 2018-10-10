import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
 
# Importing Training Set
dataset_train = pd.read_csv('AAPLTRAIN.csv')
cols = list(dataset_train)[6:11]
 
#Preprocess data
dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0,len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(",","")
 
dataset_train = dataset_train.astype(float)
 
 
training_set = dataset_train.as_matrix() # Using multiple predictors (log returns and volume)
 
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
 
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
sc_predict = MinMaxScaler(feature_range=(0,1))
sc_predict.fit_transform(training_set[:,0:1])
 
X_train = []
y_train = []
 
n_future = 3  # Number of days you want to predict into the future
n_past = 20  # Number of past days you want to use to predict the future
 
for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:5])
    y_train.append(training_set_scaled[i+n_future-1:i + n_future, 0])
 
X_train, y_train = np.array(X_train), np.array(y_train)
 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
 
# Initialize RNN
regressor = Sequential()
 
# Adding fist LSTM layer and Drop out Regularization
regressor.add(LSTM(units=10, return_sequences=True, input_shape=(n_past, 5)))
regressor.add(Dropout(.2))
 
# Adding 2nd layer with some drop out regularization
regressor.add(LSTM(units=8, return_sequences=False))
regressor.add(Dropout(.2))
 
# Output layer
regressor.add(Dense(units=1, activation='sigmoid'))
 
# Compiling
regressor.compile(optimizer='adam', loss="binary_crossentropy")  # Can change loss to mean-squared-error if you require.
 
# Fitting RNN 
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(X_train, y_train, shuffle=True, epochs=200,
                        callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
 
 
# Predictiion
dataset_test = pd.read_csv('AAPLTEST.csv')
y_true = np.array(dataset_test['log RC'])
predictions = regressor.predict(X_train[-20:])
 
predictions_to_compare = predictions
y_pred = sc_predict.inverse_transform(predictions_to_compare)


hfm, = plt.plot(y_pred, 'r', label='predicted_stock_price')
hfm2, = plt.plot(y_true,'b', label = 'actual_stock_price')

plt.legend(handles=[hfm,hfm2])
plt.title('Predictions and Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Future')
plt.savefig('graph.png', bbox_inches='tight')
plt.show()
plt.close()


hfm, = plt.plot(sc_predict.inverse_transform(y_train), 'r', label='actual_training_stock_price')
hfm2, = plt.plot(sc_predict.inverse_transform(regressor.predict(X_train)),'b', label = 'predicted_training_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions vs Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Training')
plt.savefig('graph_training.png', bbox_inches='tight')
plt.show()
plt.close()