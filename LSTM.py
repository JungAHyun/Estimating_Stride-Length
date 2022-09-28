
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler , RobustScaler 
import tensorflow.keras.regularizers as reg


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
   
    scaler = MinMaxScaler()
    raw_data =  scaler.fit_transform(df.iloc[:, 3:127])
    gait_data = scaler.fit_transform(df.iloc[:,127:139])
    stride_length_data = df.iloc[:, 139:140].to_numpy()

    raw_data= np.split(raw_data, len(raw_data))
    gait_data = np.split(gait_data, len(gait_data))
    stride_length_data = np.split(stride_length_data, len(stride_length_data))



    raw_data_list = []
    gait_data_list = []
    stride_length_data_list = []


    for i in range(len(gait_data)):
        gait_data_list.append(gait_data[i][0])
        raw_data_list.append(raw_data[i][0])
        stride_length_data_list.append(stride_length_data[i][0])
    
    input_data = []
    for i in range(len(gait_data_list)):
      temp = np.append(raw_data[i][0], gait_data[i])
      input_data.append(temp)

    return np.array(input_data), np.array(stride_length_data_list)



if __name__ == '__main__':
    gait_data, stride_length_data = get_csv()


    gait_data = gait_data.reshape(-1,136,1)
    stride_length_data.squeeze()
    
    x_train, x_test, y_train, y_test = train_test_split(gait_data, stride_length_data, test_size=0.2, shuffle=False)    
    
    model = tf.keras.Sequential()
    model.add(Conv1D(filters=64, kernel_size=2,padding='same', activation='elu', input_shape=(136,1)))
    model.add(Conv1D(filters=128, kernel_size=8, padding='same', activation='elu')) #입력데이터 한줄에 136개, 총 558줄
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=128, kernel_size=6, padding='same', activation='elu'))
    model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='elu'))
    model.add(LSTM(50, activation='elu', recurrent_activation= 'hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(10, activation='elu', recurrent_activation= 'hard_sigmoid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='elu'))
    model.add(Dense(1024, activation='elu'))
    model.add(Dense(1024, activation='elu'))
    model.add(Dense(1, activation=None))
    model.summary()

    adam= tf.keras.optimizers.Adam(lr=0.0001)
    
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping=EarlyStopping(monitor='loss',patience=5,mode='auto')

    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=100, shuffle=False, batch_size = 16, callbacks=[early_stopping])

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







