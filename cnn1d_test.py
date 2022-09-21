
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout, SimpleRNN, BatchNormalization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

path = '(input_data)Stride_length_training_test_dataset_for_CNN.xlsx'
ver = 'Ver.1 (빈칸 0 처리)'
def get_csv():
    df = pd.read_excel(path, sheet_name=ver , skiprows = 1)  #앞의 2행 생략
    gait_data =  df.iloc[:, 3:139].to_numpy()
    stride_length_data = df.iloc[:, 139:140].to_numpy()

    gait_data = np.split(gait_data, len(gait_data))
    stride_length_data = np.split(stride_length_data, len(stride_length_data))

    gait_data_list = []
    stride_length_data_list = []
    for i in range(len(gait_data)):
        gait_data_list.append(gait_data[i][0])
        stride_length_data_list.append(stride_length_data[i][0])

    print(len(gait_data_list[0]))
    print(len(stride_length_data_list))

    return np.array(gait_data_list), np.array(stride_length_data_list)


if __name__ == '__main__':
    gait_data, stride_length_data = get_csv()


    gait_data = gait_data.reshape(-1,136,1)
    stride_length_data.squeeze()
    print(gait_data.shape)
    x_train, x_test, y_train, y_test = train_test_split(gait_data, stride_length_data, test_size=0.2, shuffle=True)    
    

    model = tf.keras.Sequential()
    model.add(Conv1D(filters=1024, kernel_size=4, padding='same', activation='elu', input_shape=(136,1))) #입력데이터 한줄에 136개, 총 558줄
    model.add(Conv1D(filters=512, kernel_size=2, padding='same', activation='elu'))
    model.add(Dense(1024, activation='elu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='elu'))
    model.add(Dense(1024, activation='elu'))
    model.add(Dense(1024, activation='elu'))
    model.add(Dense(1024, activation='elu'))
    model.add(Dense(1, activation=None))



    #RNN
    # model.add(SimpleRNN(1024, input_shape=(136,1), return_sequences=True))
    # model.add(SimpleRNN(512, input_shape=(136,1), return_sequences=True))
    # model.add(SimpleRNN(256, input_shape=(136,1), return_sequences=True))
    # model.add(SimpleRNN(128, input_shape=(136,1), return_sequences=True))
    # model.add(Flatten())
    # model.add(Dense(136, activation='relu'))
    # model.add(Dense(136, activation='relu'))
    # model.add(Dense(136, activation='relu'))
    # model.add(Dense(136, activation='relu'))
    # model.add(Dense(1, activation=None))


    model.summary()
    adam= tf.keras.optimizers.Adagrad(lr=0.001)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    model.fit(x_train, y_train, epochs=500, shuffle=True, batch_size = 16)

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







