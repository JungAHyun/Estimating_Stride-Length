from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import csv   

name = '공예은'
path = 'Stride_length_experiment_stride_count_feature_extraction.xlsx'
 



def get_csv(usecols):
    df = pd.read_excel(path, sheet_name=name , usecols = [usecols, usecols+1], skiprows = 1)  #앞의 2행 생략
    left_data = []
    right_data = []
    for i in range(len(df.values)):
        left_data.append(df.values[i][0])
        right_data.append(df.values[i][1])
    
    return left_data, right_data


# 

def peak_detection(left_data, right_data):
    left_peaks, _ = find_peaks(left_data, distance=25)
    right_peaks, _ = find_peaks(right_data,  distance=25)

    
    if left_peaks[1] - left_peaks[0] > 50:
        left_peaks = np.delete(left_peaks, 0)
    elif left_peaks[-1] - left_peaks[-2] > 50:
        left_peaks = np.delete(left_peaks, -1)
    

    if right_peaks[1] - right_peaks[0] > 50:
        right_peaks = np.delete(right_peaks, 0)
    elif right_peaks[-1] - right_peaks[-2] > 50:
        right_peaks = np.delete(right_peaks, -1)
    
    
    # peak 그래프 확인
    # print('left', left_peaks)
    # print('right', right_peaks)

    # plt.plot(left_data)
    # plt.plot(left_peaks, left_data[left_peaks], "*")

    # plt.plot(right_data)
    # plt.plot(right_peaks, right_data[right_peaks], "*")

    # plt.show()

    return left_peaks, right_peaks


def heel_contact_detection(left_data, right_data):
    index = 0
    left_heel_contect = []
    right_heel_contect = []
    while(index+5<len(left_data)):
        if (left_data[index] - left_data[index+5] < 0) and (left_data[index]!=0) and (left_data[index] - left_data[index+1] < 0):
            left_heel_contect.append(index+1)
            index += 70
        else:
            index+=1
    
    index = 0
    while(index+5<len(right_data)):
        if (right_data[index] - right_data[index+5] < 0) and (right_data[index]!=0) and (right_data[index] - right_data[index+1] < 0):
            right_heel_contect.append(index+1)
            index += 70
        else:
            index+=1

    return left_heel_contect, right_heel_contect
        


def toe_off_detection(left_data, right_data):
    index = 0
    left_toe_off = []
    right_toe_off = []
    toe_off_cnt = 0

    while(index+5<len(left_data)):
        if (left_data[index]+left_data[index+1]+left_data[index+2]+left_data[index+3] == 0) and toe_off_cnt == 0:
            left_toe_off.append(index+1)
            toe_off_cnt = 1
            index+=40
        if left_data[index] > 50 and toe_off_cnt == 1:
            toe_off_cnt = 0

        index+=1
    
    index = 0    
    toe_off_cnt = 0   
    while(index+5<len(right_data)):
        if (right_data[index]+right_data[index+1]+right_data[index+2]+right_data[index+3] == 0) and toe_off_cnt == 0:
            right_toe_off.append(index+1)
            toe_off_cnt = 1
            index+=40
        if right_data[index] > 50 and toe_off_cnt == 1:
            toe_off_cnt = 0

        index+=1
    

    return left_toe_off, right_toe_off



   
        
    



if __name__ == '__main__':

    for i in range(0,24,2):
        usecols = i

        left_data, right_data = get_csv(usecols)

        # left_peaks, right_peaks= peak_detection( np.array(left_data),np.array(right_data))
        # left_heel_contect, right_heel_contect = heel_contact_detection(left_data, right_data)
        left_toe_off, right_toe_off = toe_off_detection(left_data, right_data)

        print(i,'번째 l : ', left_toe_off)
        print(i,'번째 r: ', right_toe_off)

    

    

    
    

    
    
    