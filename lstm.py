import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import random

seed = 111

random.seed(seed)
np.random.seed(seed)

# Parameters
L_cols = [['Open', 'High', 'Low', 'Close', 'Volume'], ['Open', 'Close', 'Volume'], ['High', 'Low', 'Close', 'Volume'], ['Close', 'Volume'], ['Close']]
L_sliding_window = [10, 15, 20,25, 30,35,40,45,50,55,60]
L_batch_size = [32, 64, 128]
L_dropout = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
L_num_LSTM_units = [25,30,35,40,45, 50,55,60,65,70, 75,80,85,90,95, 100]

cols = ['Close', 'Volume']
sliding_window = 0
batch_size = 0
dropout = 0
num_LSTM_units = 0

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

#Hyperparameter Initialitation
num_run = 1
num_iter = 10
num_bit_sol = 5
num_searcher = 2
num_region = 2
num_sample = 2
num_player = 4

num_identity_bit = 0
best_obj_value = 0.0
V_ij = []

searcher_sols_number = []
searcher_sol = []
sample_sol = []
sample_sol_best = []
sampleV_sol = []
searcher_sol_fitness = []
sample_sol_fitness = []
sample_sol_best_finess = []
sampleV_sol_fitness = []
best_sol = []
searcher_region_id = []
identity_bit = []
sample_sols_number = []
region_it = []
region_hl = []
expected_value = []
T_j = []
M_j = []
V_ij = []

model = Sequential()

n_sliding_window = 0
n_lstm_neuron = 0
n_dropout = 0
n_batch_size = 0
n_cols = 0

L_best = []


def rrandom(lb, ub):
    return int(lb+random.uniform(0,1)*(ub-lb))

def random_hyperparameter():
  return np.array([rrandom(0, n_sliding_window), 
                    rrandom(0, n_lstm_neuron), 
                    rrandom(0, n_dropout),
                    rrandom(0, n_batch_size),
                   rrandom(0, n_cols])

def create_hyperparameter(sol):
  
    #print("solution try", L_sliding_window[5],L_lstm_neuron[13],L_dropout[10])

    param = {
            "sliding_window" : L_sliding_window[sol[num_identity_bit]],
            "lstm_neuron" : L_num_LSTM_units[sol[num_identity_bit+1]],
            "dropout" : L_dropout[sol[num_identity_bit+2]],
            "batch_size" : L_batch_size[sol[num_identity_bit+3]],
            "column" : L_cols[sol[num_identity_bit+4]]
        }
    return param

def init():

    global searcher_sols_number, searcher_sol, sample_sol, sample_sol_best, sampleV_sol,num_identity_bit
    global searcher_sol_fitness, sample_sol_fitness, sample_sol_best_fitness, sampleV_sol_fitness, best_sol

    num_identity_bit = int(math.log2(num_region))

    searcher_sols_number = np.zeros((num_searcher, num_bit_sol+num_identity_bit),dtype=int)
    searcher_sol = np.zeros((num_searcher, num_bit_sol),dtype=int)
    sample_sol = np.zeros((num_region,num_sample,num_bit_sol+num_identity_bit),dtype=int)
    sample_sol_best = np.zeros((num_region,num_bit_sol+num_identity_bit),dtype=int) 
    sampleV_sol = np.zeros((num_searcher,num_region,num_sample*2,num_bit_sol+num_identity_bit),dtype=int)


    searcher_sol_fitness = np.zeros(num_searcher,dtype=float)
    sample_sol_fitness = np.zeros((num_region,num_sample),dtype=float)
    sample_sol_best_fitness = np.zeros(num_region,dtype=float)
    sampleV_sol_fitness = np.zeros((num_searcher,num_region,num_sample*2),dtype=float)

    best_sol = np.zeros(num_bit_sol+num_identity_bit,dtype=int)


    #searcher_sols_number = [random_hyperparameter() for i in range(num_searcher)]
    # searcher_sol = [create_hyperparameter(sols[i]) for i in range(num_searcher)]
    #print(searcher_sol)


def resource_arrangement():
    global num_identity_bit, searcher_region_id,identity_bit,searcher_sols_number,sample_sols_number,sample_sol
    global region_it, region_hl, T_j, M_j, V_ij, expected_value

    
    searcher_region_id = np.zeros(num_searcher,dtype=int)
    identity_bit = np.zeros((num_region,num_identity_bit),dtype=int)

    #print("id",num_identity_bit)
    for i in range(num_region):
      n = i
      j = num_identity_bit
      while int(n) > 0:
        j -= 1
        identity_bit[i][j] = n % 2
        n = n/2
       
    # 2.1.1 assign searcher to its region and their investment
    for i in range(num_searcher):
        region_idx = i % num_region
        searcher_region_id[i] = region_idx
        for j in range(num_identity_bit):
            searcher_sols_number[i][j] = identity_bit[region_idx][j]
        for j in range(num_identity_bit,num_bit_sol+num_identity_bit):
            searcher_sols_number[i][j] = [random_hyperparameter() for i in range(num_searcher)][i][j-num_identity_bit]
    #print("ss",searcher_sols_number)
        
    # sample_sols_number = [random_hyperparameter() for i in range(num_sample)]
    #sample_sols_number_hyperparameter = [create_hyperparameter(sample_sols_number[i]) for i in range(num_sample)]
    #print("sample solution",sample_sols_number_hyperparameter[0][0])


    #initiallize sample solution
    for i in range(num_region):
        sample_sols_number = [random_hyperparameter() for i in range(num_sample)]
        for j in range(num_sample):
            for k in range(num_identity_bit):
                sample_sol[i][j][k] = identity_bit[i][k]
            for k in range(num_identity_bit,num_bit_sol+num_identity_bit):
                sample_sol[i][j][k] = sample_sols_number[j][k-num_identity_bit]

    region_it = np.zeros(num_region,dtype=float)
    region_hl = np.ones(num_region,dtype=float)
    for i in range(num_searcher):
        idx = searcher_region_id[i]
        region_it[idx] += 1
        region_hl[idx] = 1.0

    expected_value = np.zeros((num_searcher,num_region),dtype=float)
    T_j = np.zeros(num_region,dtype=float)
    V_ij = np.zeros((num_searcher,num_region),dtype=float)
    M_j = np.zeros(num_region,dtype=float)


def vision_search():
    global sampleV_sol, searcher_sol_fitness, sample_sol_fitness, sample_sol_best_finess,sample_sol_best
    global M_j, V_ij, T_j, sampleV_sol_fitness, expected_value, sample_sol, region_hl, region_it
    global searcher_sols_number


    print("id_bit :",identity_bit)
    for i in range(num_searcher):
        for j in range(num_region):
            for k in range(num_sample):
                crossover_point = (random.randint(0,1000) % num_bit_sol) + 1
                #print("cross",crossover_point)
                m = k << 1
                for l in range(num_identity_bit):
                    sampleV_sol[i][j][m][l] = identity_bit[j][l]
                    sampleV_sol[i][j][m+1][l] = identity_bit[j][l]

                print("sample1",sampleV_sol)

                for l in range(num_identity_bit,num_bit_sol+num_identity_bit):
                    if l < crossover_point:
                        sampleV_sol[i][j][m][l] = searcher_sols_number[i][l]
                        sampleV_sol[i][j][m+1][l] = sample_sol[j][k][l]
                    else:
                        sampleV_sol[i][j][m][l] = sample_sol[j][k][l]
                        sampleV_sol[i][j][m+1][l] = searcher_sols_number[i][l]

    for i in range(num_searcher):
      for j in range(num_region):
          for k in range(2*num_sample):
              m = random.randint(0,150) % (num_bit_sol + num_identity_bit)
              if m >= num_identity_bit:
                sampleV_sol[i][j][k][m] = sampleV_sol[i][j][random.randint(0,(2*num_sample)-1)][m] 

    print("sampleV sol",sampleV_sol[0])
    print("sampleV sol",sampleV_sol[1])
    # print("sampleV sol",sampleV_sol[0][0][0])
    # print("sampleV sol",sampleV_sol[0][0][1])
    # print("sampleV sol",sampleV_sol[0][0][2])
    # print("sampleV sol",sampleV_sol[0][0][3])

    print("evaluate fitness of searcher")

    for i in range(num_searcher):
        searcher_sol_fitness[i] = evaluate_fitness(searcher_sols_number[i])

    all_sample_fitness = 0.0
    for i in range(num_region):
        rbj = sample_sol_best_fitness[i]
        idx = -1

        print("evaluate fitness of sample")
        for j in range(num_sample):
            sample_sol_fitness[i][j] = evaluate_fitness(sample_sol[i][j])
            all_sample_fitness += sample_sol_fitness[i][j]

            if sample_sol_fitness[i][j] > rbj:
                idx = j
                rbj = sample_sol_fitness[i][j]

        if idx >= 0 :
            sample_sol_best_fitness[i] = rbj
            sample_sol_best[i] = sample_sol[i][idx]

    print("searcher_fitness : ",searcher_sol_fitness)
    # M_j
    for i in range(num_region):
        M_j[i] = sample_sol_best_fitness[i] / all_sample_fitness

    print("evaluate fitness of sampleV")

    for i in range(num_searcher):
        for j in range(num_region):
            V_ij[i][j] = 0.0
            for k in range(num_sample):
                m = k << 1
                sampleV_sol_fitness[i][j][m] = evaluate_fitness(sampleV_sol[i][j][m])
                sampleV_sol_fitness[i][j][m+1] = evaluate_fitness(sampleV_sol[i][j][m+1])
                V_ij[i][j] += sampleV_sol_fitness[i][j][m] + sampleV_sol_fitness[i][j][m+1]
            V_ij[i][j] =  2*num_sample

            
    for i in range(num_region):
        T_j[i] = region_it[i] / region_hl[i]


    for i in range(num_searcher):
        for j in range(num_region):
            expected_value[i][j] = T_j[i] * V_ij[i][j] * M_j[j]

    
    for i in range(num_searcher):
        for j in range(num_region):
            for k in range(num_sample):
                m = k << 1
                if sampleV_sol_fitness[i][j][m] > sample_sol_fitness[j][k]:
                    for l in range(num_identity_bit,num_bit_sol):
                        sample_sol[j][k][l] = sampleV_sol[i][j][m][l]
                    sample_sol_fitness[j][k] = sampleV_sol_fitness[i][j][m]
                
                if sampleV_sol_fitness[i][j][m+1] > sample_sol_fitness[j][k]:
                    for l in range(num_identity_bit,num_bit_sol):
                        sample_sol[j][k][l] = sampleV_sol[i][j][m+1][l]
                    sample_sol_fitness[j][k] = sampleV_sol_fitness[i][j][m+1]

    for i in range(num_region):
        region_hl[i] += 1

    for i in range(num_searcher):
        play0_idx = random.randint(0,1000) % num_region
        play0_ev = expected_value[i][play0_idx]

        for j in range(num_player-1):
            play1_idx = random.randint(0,1000) % num_region
            if expected_value[i][play1_idx] > play0_ev:
                play0_idx = play1_idx
                play0_ev = expected_value[i][play0_idx]

        for j in range(num_sample):
            if sample_sol_fitness[play0_idx][j] > searcher_sol_fitness[i]:
                searcher_sols_number[i] = sample_sol[play0_idx][j]
                searcher_sol_fitness[i] = sample_sol_fitness[play0_idx][j]

        region_it[play0_idx] += 1
        region_hl[play0_idx] = 1

def marketing_survey():
    global best_obj_value, region_it,best_obj_value,best_sol

    max_obj_value = []
    max_best_sol = []

    for i in range(num_region):
        if region_hl[i] > 1:
            region_it[i] = 1.0

    
    j = -1
    for i in range(num_searcher):
        if searcher_sol_fitness[i] > best_obj_value:
            best_obj_value = searcher_sol_fitness[i]
            j = i
        

    if j >= 0:
        best_sol = searcher_sols_number[j]


    max_obj_value.append(best_obj_value)
    max_best_sol.append(best_sol)
    print("Current best solution",best_sol)

    global L_best
    L_best.append(best_obj_value)


def evaluate_fitness(sol):
    global model,history, sliding_window,lstm_neuron,dropout,cols

    hyp = create_hyperparameter(sol)

    sliding_window = int(hyp['sliding_window'])
    num_LSTM_units = int(hyp['lstm_neuron'])
    dropout = float(hyp['dropout'])
    batch_size = int(hyp['batch_size'])
    cols = hyp['columns']


    # print("Parameter : Sliding window = ",sliding_window," lstm_neuron = ",num_LSTM_units," dropout = ",dropout, " batch_size :",batch_size)

    fit_value = 0
    fit_value = runLSTM(sliding_window, batch_size, dropout, num_LSTM_units, num_epochs)

    return fit_value

def main():

    global n_sliding_window,n_lstm_neuron,n_dropout,n_batch_size,n_cols
    n_sliding_window = len(L_sliding_window)
    n_lstm_neuron = len(L_num_LSTM_units)
    n_dropout = len(L_dropout)
    n_batch_size = len(L_batch_size)
    n_cols = len(L_cols)
    

    # Importing Data
    df = pd.read_csv('https://www.dropbox.com/s/rvhoqedvj4yir9w/BBCAJK.csv?dl=1')
    df = df.dropna()

    dataset, out = preprocessData(df)

    threshold1 = 0.8
    threshold2 = 0.9

    

    dataset = x_scaler.fit_transform(dataset)
    out = y_scaler.fit_transform(out)

    best = 0

    counter = 0

    train, valid, test, out_train, out_valid, out_test = splitData(dataset, out, sliding_window,
                                                                   threshold1, threshold2)

    scaleData(train, valid, test, out_train, out_valid, out_test)

    avg_obj_value_iter = np.zeros(num_iter, dtype=float)
    for i in range(num_run):
        init()                 
        resource_arrangement() 
        for j in range(num_iter):
            vision_search()
            marketing_survey()
            avg_obj_value_iter[j] += best_obj_value
    
    for i in range(num_iter):
        print('Iteration :', i , 'result' , avg_obj_value_iter[i]/num_run)
        
    print("best solution",best_sol)
    best_hyp = create_hyperparameter(best_sol)
    print("best_sol_parameter",best_hyp)

    f = open("result.txt", "w+")
    for i in range(num_iter):
      f.write("{}, {}".format(i, avg_obj_value_iter[i]/num_run))
    f.close()

    f = open("best.txt", "w+")
    f.write("{}\n{}".format(best_sol,best_hyp))
    f.close()
    # plt.figure(figsize=(10, 6))
    # plt.plot(L_best, color='black', label='Best Current RMSE')
    # plt.title('Convergence')
    # plt.xlabel('Iteration')
    # plt.ylabel('RMSE')
    # plt.legend()
    # plt.show()

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

    # r2 = r2_score(inv_y, inv_y_hat)

    # print("MSE: {}, RMSE: {}".format(mse, rmse))

    # return rmse, r2

    fit_value = 1 / rmse

    return fit_value



main()

