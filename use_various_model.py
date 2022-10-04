

from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout, LSTM, Conv2D, MaxPooling2D, SimpleRNN, GRU
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow.keras.regularizers as reg
from sklearn.preprocessing import MinMaxScaler, StandardScaler , RobustScaler
from tensorflow.keras.layers import BatchNormalization as BN


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

x_train_data_path = 'data_file\\save_dataset\\x_train_dataset.xlsx'
x_test_data_path = 'data_file\\save_dataset\\x_test_dataset.xlsx'
y_train_data_path = 'data_file\\save_dataset\\y_train_dataset.xlsx'
y_test_data_path = 'data_file\\save_dataset\\y_test_dataset.xlsx'

input_dataset =[]

def get_csv():
    global input_dataset

    x_train_df = pd.read_excel(x_train_data_path, skiprows = 0)  
    x_test_df = pd.read_excel(x_test_data_path, skiprows = 0)  
    y_train_df = pd.read_excel(y_train_data_path, skiprows = 0)  
    y_test_df = pd.read_excel(y_test_data_path, skiprows = 0)  

    x_train_data = x_train_df.iloc[:, 1:].to_numpy()
    x_test_data = x_test_df.iloc[:, 1:].to_numpy()
    y_train_data = y_train_df.iloc[:, 1:].to_numpy()
    y_test_data = y_test_df.iloc[:, 1:].to_numpy()

    return np.array(x_train_data), np.array(x_test_data) ,np.array(y_train_data), np.array(y_test_data)


def make_GRU(model):
    model.add(GRU(8, activation='elu', recurren_activation= 'hard_sigmoid', return_sequences=True, input_shape=(375,1) ))
    model.add(Dropout(0.01))
    model.add(GRU(4, activation='elu', recurrent_activation= 'hard_sigmoid'))
    model.add(Dropout(0.01))
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1, activation=None))
    
    return model 


def make_rnn(model):
    model = tf.keras.Sequential()
    model.add(SimpleRNN(512,activation='selu', input_shape=(375,1), return_sequences=True))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(SimpleRNN(512,  activation='selu', return_sequences=True)) #입력데이터 한줄에 136개, 총 558줄
    model.add(SimpleRNN(512,  activation='selu', return_sequences=True))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(SimpleRNN(512, activation='selu', return_sequences=True))
    model.add(SimpleRNN(512, activation='selu', return_sequences=True))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1, activation=None))
    
    return model 

def make_1d_cnn(model):
    model = tf.keras.Sequential()
    model.add(Conv1D(filters=512, kernel_size=2,padding='same', activation='selu', input_shape=(375,1)))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=512, kernel_size=8, padding='same', activation='selu')) #입력데이터 한줄에 136개, 총 558줄
    model.add(Conv1D(filters=512, kernel_size=6, padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='selu'))
    model.add(Conv1D(filters=512, kernel_size=2, padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1, activation=None))

    return model


def make_1d_cnn_lstm(model):
    model.add(Conv1D(filters=512, kernel_size=2,padding='same', activation='selu', input_shape=(375,1)))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=512, kernel_size=8, padding='same', activation='selu')) #입력데이터 한줄에 136개, 총 558줄
    model.add(Conv1D(filters=512, kernel_size=6, padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='selu'))
    model.add(Conv1D(filters=512, kernel_size=2, padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(LSTM(16, activation='elu', recurrent_activation= 'hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.01))
    model.add(LSTM(8 activation='elu', recurrent_activation= 'hard_sigmoid'))
    model.add(Dropout(0.01))
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1, activation=None))

    return model 


def make_lstm(model):
    model.add(LSTM(16, activation='elu', recurrent_activation= 'hard_sigmoid', return_sequences=True, input_shape=(375,1) ))
    model.add(Dropout(0.01))
    model.add(LSTM(8, activation='elu', recurrent_activation= 'hard_sigmoid'))
    model.add(Dropout(0.01))
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1, activation=None))
    
    return model

def make_2d_cnn(model):
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='selu', input_shape = (3,125,1))) #입력데이터 한줄에 125개, 총 558줄
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='selu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling2D(1,2))
    model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same', activation='selu'))
    model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling2D(1,2))
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1024, activation='selu'))
    model.add(BN())
    model.add(Dense(1, activation=None))

    return model 


if __name__ == '__main__':

    x_train, x_test, y_train,y_test  = get_csv()
    print(x_train.shape)   

    x_train = x_train.reshape(-1,375,1)
    y_train.squeeze()

    print(x_train[0])
    print(x_train.shape)    #(446, )
    print(y_train.shape)    #(446, 375, 1)

    # x_train, x_test, y_train, y_test = train_test_split(gait_data, stride_length_data, test_size=0.2, shuffle=True)    
  
    model = tf.keras.Sequential()

    #LSTM 
    model = make_lstm(model)

    #CNN-LSTM
    model=make_1d_cnn_lstm(model)

    # #RNN 
    # model = make_rnn(model)
    
    #GRU
    # model = make_GRU(model)

    #1D-CNN
    # model = make_1d_cnn(model)

    #2D-CNN
    # model =make _2d_cnn(model)

    model.summary()
    adam= tf.keras.optimizers.Adam(lr=0.0001)
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=20, shuffle=False, batch_size = 16)

    print(x_test.shape)
    x_test = x_test.reshape(-1,375,1)
    y_test.squeeze()
    print('x_test: ', len(x_test))

    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()

    y_testdata = []

    for i in range(len(y_test)):
        y_testdata.append(y_test[i][0])

    
    print(len(y_testdata))
    print(len(y_pred))

    print("save end")

    # print("======= ",mode ," report ==========")
    print("MAE: ",mean_absolute_error(y_true=y_testdata, y_pred=y_pred))
    print("MAE2: ", np.mean(np.abs(y_testdata - y_pred)))
    print("std: ", np.std(np.absolute(np.subtract(y_testdata, y_pred))))
    print("=========================")
    print("ME: ", np.mean(np.subtract(y_testdata, y_pred)))
    print("std: ", np.std(np.subtract(y_testdata, y_pred)))
    print("=========================")
    print("mean relative error: ", np.mean(np.divide(np.absolute(np.subtract(y_testdata, y_pred)), y_testdata))*100)
    print("=========================")        
    print("r2 score: ", r2_score(y_true=y_testdata, y_pred=y_pred))
    print("=========================")
    
    from matplotlib import pyplot as plt
    plt.plot(y_pred, 'g')
    plt.plot(y_testdata, 'r')
    plt.show()

    plt.xlabel('y_test')
    plt.ylabel('y_pred')

    fit_line = np.polyfit(y_testdata, y_pred, 1)
    x_minmax = np.array([min(y_testdata), max(y_testdata)]) # x축 최소값, 최대값

    fit_y = x_minmax * fit_line[0] + fit_line[1]

    plt.scatter(y_testdata, y_pred, color = 'r', s = 20)
    plt.plot(x_minmax, fit_y, color = 'orange') # 회귀선 그래프 그리기
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()






