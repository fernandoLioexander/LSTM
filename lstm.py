import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.constraints import max_norm
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from numba import jit, cuda 

@jit(target ="cuda") 

# Parameters
L_cols = [['Open', 'High', 'Low', 'Close', 'Volume'], ['Open', 'Close', 'Volume'], ['High', 'Low', 'Close', 'Volume'], ['Close', 'Volume'], ['Close']]
L_sliding_window = [5, 10, 15, 20, 30, 60]
L_batch_size = [32, 64, 128]
L_dropout = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
L_num_LSTM_units = [25, 50, 75, 100]

cols = ['Close', 'Volume']
sliding_window = 30
batch_size = 30
dropout = 0.5
num_LSTM_units = 50

num_epochs = 25
num_features = len(cols)

# Containers for Preprocessed Data
scaled_train = []
scaled_valid = []
scaled_test = []
scaled_out_train = []
scaled_out_valid = []
scaled_out_test = []

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

real_price = []

B_cols = []
B_sliding_window = 0
B_batch_size = 0
B_dropout = 0
B_num_LSTM_units = 0
L_fitness = []

def main():
    # Importing Data
    df = pd.read_csv("C:/Users/Daniel/Documents/Skripsi/BBCA.JK2.csv")
    df = df.dropna()

    dataset, out = preprocessData(df)

    threshold1 = 0.8
    threshold2 = 0.9

    dataset = x_scaler.fit_transform(dataset)
    out = y_scaler.fit_transform(out)

    best = 0

    counter = 0

    # for i in L_cols:
    #     for j in L_num_LSTM_units:
    #         for k in L_sliding_window:
    #             for l in L_batch_size:
    #                 for m in L_dropout:
    #                     for n in range(3):
    #                         counter = counter + 1
    #
    #                         global cols
    #                         cols = i
    #                         global num_LSTM_units
    #                         num_LSTM_units = j
    #                         global sliding_window
    #                         sliding_window = k
    #                         global batch_size
    #                         batch_size = l
    #                         global dropout
    #                         dropout = m
    #
    #                         train, valid, test, out_train, out_valid, out_test = splitData(dataset, out, sliding_window,
    #                                                                                        threshold1, threshold2)
    #
    #                         scaleData(train, valid, test, out_train, out_valid, out_test)
    #
    #                         rmse, r2 = runLSTM(sliding_window, batch_size, dropout, num_LSTM_units, num_epochs)
    #
    #                         fitness = 1 / rmse
    #
    #                         print("Counter: {}, RMSE: {}".format(counter,rmse))
    #
    #                         global L_fitness
    #                         L_fitness.append(fitness)
    #
    #                         if fitness > best:
    #                             best = fitness
    #
    #                             global B_cols
    #                             B_cols = cols
    #                             global B_sliding_window
    #                             B_sliding_window = sliding_window
    #                             global B_batch_size
    #                             B_batch_size = batch_size
    #                             global B_dropout
    #                             B_dropout = dropout
    #                             global B_num_LSTM_units
    #                             B_num_LSTM_units = num_LSTM_units
    #
    # print("Best: {}".format(best))
    # print("cols: {}".format(B_cols))
    # print("sliding_window: {}".format(B_sliding_window))
    # print("batch_size: {}".format(B_batch_size))
    # print("dropout: {}".format(B_dropout))
    # print("num_LSTM_units: {}".format(B_num_LSTM_units))
    #
    # with open("fitness_results.txt", "w") as f:
    #     for item in L_fitness:
    #         f.write("{}\n".format(item))

    train, valid, test, out_train, out_valid, out_test = splitData(dataset, out, sliding_window,
                                                                   threshold1, threshold2)

    scaleData(train, valid, test, out_train, out_valid, out_test)

    rmse, r2 = runLSTM(sliding_window, batch_size, dropout, num_LSTM_units, num_epochs)

    # print("RMSE: {}".format(rmse))
    # print("R2: {}".format(r2))

def preprocessData(df):
    # Selecting Columns
    dataset = pd.DataFrame(index=range(0, len(df)), columns=cols)
    j = 0
    for i in range(0, len(df)):
        valid = 0
        while valid == 0:
            try:
                for x in range(0, num_features):
                    dataset[cols[x]][i] = df[cols[x]][j]
                j = j + 1
                valid = 1
            except:
                j = j + 1

    out = dataset['Close']
    out = np.array(out)
    out = out.reshape(-1, 1)

    global real_price
    real_price = out

    return dataset, out

def splitData(dataset, out, sliding_window, threshold1, threshold2):
    # Splitting Data into Train & Test
    split1 = math.ceil(threshold1 * len(dataset))
    split2 = math.ceil(threshold2 * len(dataset))

    train = dataset[:split1]
    valid = dataset[split1 - sliding_window:split2]
    test = dataset[split2 - sliding_window:]

    out_train = out[:split1]
    out_valid = out[split1 - sliding_window:split2]
    out_test = out[split2 - sliding_window:]

    return train, valid, test, out_train, out_valid, out_test

def scaleData(train, valid, test, out_train, out_valid, out_test):
    # Scaling Data
    global scaled_train
    scaled_train = train
    global scaled_valid
    scaled_valid = valid
    global scaled_test
    scaled_test = test

    global scaled_out_train
    scaled_out_train = out_train
    global scaled_out_valid
    scaled_out_valid = out_valid
    global scaled_out_test
    scaled_out_test = out_test

def runLSTM(sliding_window, batch_size, dropout, num_LSTM_units, num_epochs):
    # Converting Training Dataset into x_train and y_train
    x_train = []
    y_train = []
    for i in range(sliding_window, len(scaled_train)):
        x_train.append(scaled_train[i - sliding_window:i])
        y_train.append(scaled_out_train[i])
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_train = np.reshape(x_train, (len(x_train), sliding_window, num_features))
    y_train = y_train.flatten()

    x_valid = []
    y_valid = []
    for i in range(sliding_window, len(scaled_valid)):
        x_valid.append(scaled_valid[i - sliding_window:i])
        y_valid.append(scaled_out_valid[i])
    x_valid = np.array(x_valid, dtype=np.float32)
    y_valid = np.array(y_valid, dtype=np.float32)
    x_valid = np.reshape(x_valid, (len(x_valid), sliding_window, num_features))
    y_valid = y_valid.flatten()

    # Create and Fit the LSTM Network
    model = Sequential()
    model.add(LSTM(units=num_LSTM_units, kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3), bias_constraint=max_norm(3), return_sequences=True, input_shape=(sliding_window, num_features)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=num_LSTM_units, kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3), bias_constraint=max_norm(3), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=num_LSTM_units, kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3), bias_constraint=max_norm(3), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=num_LSTM_units, kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=0, validation_data=(x_valid, y_valid))

    # Preparing Test Dataset
    x_test = []
    y_test = []
    for i in range(sliding_window, len(scaled_test)):
        x_test.append(scaled_test[i - sliding_window:i])
        y_test.append(scaled_out_test[i])
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    x_test = np.reshape(x_test, (len(x_test), sliding_window, num_features))

    # Predicting Values Using Past Data
    y_hat = model.predict(x_test)
    inv_y_hat = y_scaler.inverse_transform(y_hat)

    inv_y = y_scaler.inverse_transform(y_test)

    mse = mean_squared_error(inv_y, inv_y_hat)

    rmse = math.sqrt(mse)

    r2 = r2_score(inv_y, inv_y_hat)

    print("MSE: {}, RMSE: {}".format(mse, rmse))

    plt.figure(figsize=(10, 6))
    plt.plot(inv_y, color='blue', label='Actual Stock Price')
    plt.plot(inv_y_hat, color='red', label='Predicted Stock Price')
    plt.title('BCA Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('BCA Stock Price')
    plt.legend()
    plt.show()

    return rmse, r2

    # fit_value = 1 / rmse

    # return fit_value



main()

