

import csv
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout, LSTM, BatchNormalization , Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_logarithmic_error as msle
from tensorflow.keras.layers import BatchNormalization as BN
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow.keras.regularizers as reg
from sklearn.preprocessing import MinMaxScaler, StandardScaler , RobustScaler 


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

input_data_path = 'data_file\\use_velocity_input_data\\(velocity, no normalization, no exist gait data)input_data.xlsx'
label_data_path = 'data_file\\use_data\\Stride_Length_Data.xlsx'

input_dataset =None

def get_csv():
  global input_dataset
  input_df = pd.read_excel(input_data_path, skiprows = 0)  #앞의 1행 생략
  label_df = pd.read_excel(label_data_path, skiprows = 0)  #앞의 1행 생략    


  input_data = input_df.iloc[:, 1:].to_numpy()
  label_data = label_df.iloc[:, :].to_numpy()

  input_data= np.split(input_data, len(input_data))
  label_data = np.split(label_data, len(label_data))

  input_data_list = []
  label_data_lsit = []

  for i in range(len(input_data)):
    input_data_list.append(input_data[i][0])
  for i in range(len(label_data)):
    label_data_lsit.append(label_data[i][0])

  print(len(input_data_list))
  print(len(label_data_lsit))

  return np.array(input_data_list), np.array(label_data_lsit)

''''
def make_input_dataset(gait_data):
  global input_dataset
  input_df = pd.read_excel(input_data_path, skiprows = 0)  #앞의 1행 생략
  label_df = pd.read_excel(label_data_path, skiprows = 0)  #앞의 1행 생략    

  f1 = []
  f2 = []
  f3 = []
  f4 = []
  for i in range(0, 2229, 4):
    f1.append(input_df.iloc[i:i+1, 1:].to_numpy())
    f2.append(input_df.iloc[i+1:i+2, 1:].to_numpy())
    f3.append(input_df.iloc[i+2:i+3, 1:].to_numpy())
    f4.append(input_df.iloc[i+3:i+4, 1:].to_numpy())
  
    f5 =  np.concatenate([f1[0], f2[0],f3[0], f4[0]], 0)
    input_dataset.append(f5)
  

  label_data = label_df.iloc[:, :].to_numpy()

#   input_data= np.split(input_data, len(input_data))
  label_data = np.split(label_data, len(label_data))

  input_data_list = []
  label_data_lsit = []

#   for i in range(len(input_data)):
#     input_data_list.append(input_data[i][0])
  for i in range(len(label_data)):
    label_data_lsit.append(label_data[i][0])

  print(len(input_data_list))
  print(len(label_data_lsit))

  return np.array(input_dataset), np.array(label_data_lsit)
'''


def save_dataset(x_train, x_test, y_train, y_test):
  input = []
  input_list=[]
  label = []
  for i in range(len(x_train)):
    for j in range(len(x_train[i])):
      input = []
      for k in range(len(x_train[i][j])):
        tmp = x_train[i][j][k]
        input.append(tmp[0])
    input_list.append(input)

  print(input_list[0])    
  

  for i in range(len(x_test)):
    tmp = x_test[0][0][0]
    input.append(tmp[0])
  
  for i in range(len(y_train)):
    label.append(y_train[i][0])
  for i in range(len(y_test)): 
    label.append(y_test[i][0])  


  print(len(input))
  f = open('shuffle input_dataset.csv', 'w', encoding='utf-8', newline='')
  wr = csv.writer(f)
  print(list(input[0]))
  for i in range(len(input)):
    wr.writerow(input[i])


  input_list = pd.DataFrame(data=list(input))
  input_list.to_csv('shuffle input_dataset.csv')

  # label_list = pd.DataFrame(data=list(label))
  # label_list.to_csv('shuffle label_dataset.csv')


  print('-------------------------- Save End --------------------------')
  exit()


if __name__ == '__main__':

  gait_data, stride_length_data = get_csv()
  # input_dataset = make_input_dataset(gait_data)
 
  gait_data = gait_data.reshape(-1,5,125,1)
  stride_length_data.squeeze()

  print(stride_length_data.shape)    #(558, 1)
  print(gait_data.shape)             #(558, 4, 125, 1)

  print(type(gait_data[0][0][0]))
  print(gait_data[0][0][0])
  x_train, x_test, y_train, y_test = train_test_split(gait_data, stride_length_data, test_size=0.2, shuffle=True) 
  print('x',x_train[0])

  save_dataset(x_train, x_test, y_train, y_test)  
  exit()
     
  

  model = tf.keras.Sequential()
  model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='selu', input_shape = (5,125,1))) #입력데이터 한줄에 125개, 총 558줄
  model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='selu'))
  model.add(BN())
  model.add(MaxPooling2D(2,2))
  model.add(Conv2D(filters=128, kernel_size=(4,4), padding='same', activation='selu'))
  model.add(Conv2D(filters=128, kernel_size=(4,4), padding='same', activation='selu'))
  model.add(BN())
  model.add(MaxPooling2D(2,2))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='selu'))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='selu'))  
  model.add(BN())
  model.add(MaxPooling2D(1,2))
  model.add(Flatten())
  model.add(Dense(1024, activation='selu'))
  model.add(BN())
  model.add(Dense(1024, activation='selu'))
  model.add(BN())
  model.add(Dense(1024, activation='selu'))
  model.add(BN())
  model.add(Dense(1, activation=None))
  model.summary()


  adam= tf.keras.optimizers.Adam(lr=0.0001)
  
      
  # from tensorflow.keras.callbacks import EarlyStopping
  # early_stopping=EarlyStopping(monitor='loss',patience=5,mode='auto')

  model.compile(optimizer=adam, loss = 'mse', metrics=['mae'])
  model.fit(x_train, y_train, epochs = 500, shuffle=False, batch_size =16) #, callbacks=[early_stopping])

  
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




