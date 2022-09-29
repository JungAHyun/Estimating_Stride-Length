import csv
from macpath import split
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout, LSTM, BatchNormalization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow.keras.regularizers as reg
from sklearn.preprocessing import MinMaxScaler, StandardScaler , RobustScaler 


gait_path = 'data_file\\(input_data)Stride_length_training_test_dataset_for_CNN.xlsx'
copx_path = 'data_file\\1_Stride_Extraction_COPx_velocity.xlsx'
copy_path = 'data_file\\1_Stride_Extraction_COPy_velocity.xlsx'

ver = 'Ver.1 (빈칸 0 처리)'
cop_sheet= '행 정리'
copx_vel_sheet = '(행)COPx_velocity'
copy_vel_sheet = '(행)COPy_velocity'

scaler = MinMaxScaler()
def get_gait_csv():
    df = pd.read_excel(gait_path, sheet_name=ver , skiprows = 1)  #앞의 2행 생략
    
    # normalization version
    raw_data =  scaler.fit_transform(df.iloc[:, 2:127])
    gait_data =  scaler.fit_transform(df.iloc[:, 127:139])
    stride_length_data =  scaler.fit_transform(df.iloc[:, 139:140])

    # no normalization version
    # raw_data = df.iloc[:, 2:127].to_numpy()
    # gait_data =  df.iloc[:, 127:139].to_numpy()
    # stride_length_data = df.iloc[:, 139:140].to_numpy()
    
    raw_data = np.split(raw_data, len(raw_data))
    gait_data = np.split(gait_data, len(gait_data))
    stride_length_data = np.split(stride_length_data, len(stride_length_data))

    raw_data_list =[]
    gait_data_list = []
    stride_length_data_list = []
    
    for i in range(len(raw_data)):
        raw_data_list.append(raw_data[i][0])
    for i in range(len(gait_data)):
        gait_data_list.append(gait_data[i][0])
    for i in range(len(stride_length_data)):
        stride_length_data_list.append(stride_length_data[i][0])

    print(len(raw_data_list))
    print(len(gait_data_list))
    print(len(stride_length_data_list))

    return raw_data_list, gait_data_list, stride_length_data_list


def get_cop_csv():
    copx_df = pd.read_excel(copx_path, sheet_name=cop_sheet , skiprows = 0)  #앞의 2행 생략
    copy_df = pd.read_excel(copy_path, sheet_name=cop_sheet , skiprows = 0)  #앞의 2행 생략

    # normalization version
    copx_data =  scaler.fit_transform(copx_df.iloc[:, 2:127])
    copy_data =  scaler.fit_transform(copy_df.iloc[:, 2:127])
   
    # no normalization version
    # copx_data =  copx_df.iloc[:, 2:127].to_numpy()
    # copy_data = copy_df.iloc[:, 2:127].to_numpy()

    # copx_data = np.split(copx_data, len(copx_data))
    # copy_data = np.split(copy_data, len(copy_data))
    # print('copx_data', len(copx_data))
    # print(copx_data[0])
    # copx_data_list = []
    # copy_data_list = []
    # for i in range(len(copx_data)):
    #     copx_data_list.append(copx_data[i][0])
    #     copy_data_list.append(copy_data[i][0])

    print(copx_data[0])
    print(copy_data[0])

    return copx_data, copy_data



def get_cop_vel_csv():
    copx_vel_df = pd.read_excel(copx_path, sheet_name=copx_vel_sheet , skiprows = 0)  #앞의 2행 생략
    copy_vel_df = pd.read_excel(copy_path, sheet_name=copy_vel_sheet , skiprows = 0)  #앞의 2행 생략

    # normalization version
    copx_vel_data =  scaler.fit_transform(copx_vel_df.iloc[:, 2:127])
    copy_vel_data =  scaler.fit_transform(copy_vel_df.iloc[:, 2:127])

    print('copx_vel_data', len(copx_data))

    # no normalization version
    # copx_vel_data =  copx_vel_df.iloc[:, 2:127].to_numpy()
    # copy_vel_data = copy_vel_df.iloc[:, 2:127].to_numpy()

    # copx_vel_data = np.split(copx_vel_data, len(copx_vel_data))
    # copy_vel_data = np.split(copy_vel_data, len(copy_vel_data))

    # copx_vel_data_list = []
    # copy_vel_data_list = []
    # for i in range(len(copx_vel_data_list)):
    #     copx_vel_data_list.append(copx_vel_data[i][0])
    #     copy_vel_data_list.append(copy_vel_data[i][0])

    print(copx_vel_data[0])
    print(copx_vel_data[0])
    return copx_vel_data, copy_vel_data





cnt = 0
def save_csv(raw, gait, copx, copy, x_vel, y_vel):
    global cnt
    zero = np.zeros(113)  
    gait_data =[]

    print(type(raw[0][0]))
    for i in range(len(raw)):
        tmp = np.concatenate([gait[i], zero])
        gait_data.append(tmp)
    
    f = open('(velocity, normalization, exist gait data)input_data.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    data_list = []
    print(len(x_vel))
    for i in range(len(raw)):
        data_list.append(list(raw[i]))
        data_list.append(list(copx[i]))
        data_list.append(list(copy[i]))
        data_list.append(list(x_vel[i]))
        data_list.append(list(y_vel[i]))
        data_list.append(list(gait_data[i]))


    data_list = pd.DataFrame(data=data_list)
    data_list.to_csv('(velocity, normalization, exist gait data)input_data.csv')
    print('-------------------------- Save End --------------------------')







if __name__ == '__main__':
    raw_data, gait_data, stride_length_data = get_gait_csv()
    copx_data, copy_data  = get_cop_csv()
    copx_vel_data, copy_vel_data  = get_cop_vel_csv()
    
    print('len',len(copx_vel_data))
    print('len',len(copy_vel_data))
    save_csv(raw_data, gait_data, copx_data, copy_data, copx_vel_data , copy_vel_data)
