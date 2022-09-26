

from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout, LSTM, BatchNormalization , Conv2D, MaxPooling2D
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
    x_test_df = pd.read_excel(x_train_data_path, skiprows = 0)  
    y_train_df = pd.read_excel(y_train_data_path, skiprows = 0)  
    y_test_df = pd.read_excel(y_test_data_path, skiprows = 0)  

    x_train_data = x_train_df.iloc[:, 1:].to_numpy()
    x_test_data = x_test_df.iloc[:, 1:].to_numpy()
    y_train_data = y_train_df.iloc[:, 1:].to_numpy()
    y_test_data = y_test_df.iloc[:, 1:].to_numpy()

  
    # input_data= np.split(input_data, len(input_data))
    # label_data = np.split(label_data, len(label_data))

    # input_data_list = []
    # label_data_lsit = []

    # for i in range(len(input_data)):
    #     input_data_list.append(input_data[i][0])
    # for i in range(len(label_data)):
    #     label_data_lsit.append(label_data[i][0])

    # print(len(input_data_list))
    # print(len(label_data_lsit))

    return np.array(x_train_data), np.array(x_test_data) ,np.array(y_train_data), np.array(y_test_data)


    


if __name__ == '__main__':

    x_train, x_test, y_train,y_test  = get_csv()
    print(x_train.shape)   

    x_train = x_train.reshape(-1,375,1)
    y_train.squeeze()


    print(x_train.shape)    #(446, )
    print(y_train.shape)    #(446, 375, 1)


    # x_train, x_test, y_train, y_test = train_test_split(gait_data, stride_length_data, test_size=0.2, shuffle=True)    

    model = tf.keras.Sequential()
    model.add(Conv1D(filters=512, kernel_size=2,padding='same', activation='selu', input_shape=(375,1)))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=256, kernel_size=8, padding='same', activation='selu')) #입력데이터 한줄에 136개, 총 558줄
    model.add(Conv1D(filters=256, kernel_size=6, padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='selu'))
    model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='selu'))
    model.add(BN())
    model.add(MaxPooling1D(2))
    model.add(Dense(1024, activation='selu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1024, activation='selu'))
    model.add(Dense(1, activation=None))
    model.summary()

    adam= tf.keras.optimizers.Adam(lr=0.0001)
    
        
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs= 300, shuffle=True, batch_size = 32)

    x_test = x_test.reshape(-1,375,1)
    y_test.squeeze()

    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()

    y_testdata = []

    for i in range(len(y_test)):
        y_testdata.append(y_test[i][0])

    
    print(y_testdata)
    print(y_pred)

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




