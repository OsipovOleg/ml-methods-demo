# %%
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 100
plt.style.use('seaborn-whitegrid')

np.set_printoptions(precision=4)

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

# %%
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


# %%


def f(t):
    return 10 * (np.sin(t) - np.cos(t) - np.sin(3 * t))


n_points = 100000
a, b = 0, 1000
t = np.linspace(a, b, n_points)
x = f(t)
plt.plot(t[:2000], x[:2000], c='r')
# %%
sample_len = 10
n_samples = n_points // sample_len
n_features = 2

# %%
time_feature = np.random.uniform(low=0, high=b, size=(n_samples, 1))
time_delta = np.zeros((n_samples, 1))
for i in range(sample_len - 1):
    r = np.random.uniform(low=0, high=1, size=(n_samples,))
    time_delta = np.append(time_delta, r.reshape(n_samples, 1), axis=1)
    temp = time_feature[:, -1] + r
    time_feature = np.append(time_feature, temp.reshape(n_samples, 1), axis=1)

time_delta = np.delete(time_delta, 0, 1)
print('time_feature =')
print(time_feature)
print('time_delta = ')
print(time_delta)

# %%
f_feature = f(time_feature)
print('f_feature')
f_feature
# %%
Y = f_feature[:, -1]
Y = np.reshape(Y, (n_samples, 1))
Y
# %%
f_feature = np.delete(f_feature, -1, 1)
f_feature

# %%
X = np.dstack((time_delta, f_feature))
X
# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# %%
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True,
               input_shape=(sample_len - 1, n_features)))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()
# %%

history = model.fit(X_train, y_train, epochs=50,
                    validation_split=0.2, verbose=1)
# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# %%
pred = model.predict(X_test)
pred
# %%
y_test
# %%
mean_squared_error(y_test, pred)

