from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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

def peak_detection(left_data, right_data):
    left_peaks, _ = find_peaks(left_data, distance=25)
    right_peaks, _ = find_peaks(right_data,  distance=25)

    
    if left_peaks[1] - left_peaks[0] > 50:
        left_peaks = np.delete(left_peaks, 0)
    elif left_peaks[-1] - left_peaks[-2] > 50:
        left_peaks = np.delete(left_peaks, -1)
    

    if right_peaks[1] - right_peaks[0] > 50:
        print('in')
        right_peaks = np.delete(right_peaks, 0)
    elif right_peaks[-1] - right_peaks[-2] > 50:
        right_peaks = np.delete(right_peaks, -1)
    
    

    print('left', left_peaks)
    print('right', right_peaks)

    plt.plot(left_data)
    plt.plot(left_peaks, left_data[left_peaks], "*")

    plt.plot(right_data)
    plt.plot(right_peaks, right_data[right_peaks], "*")

    plt.show()

    return left_peaks, right_peaks




if __name__ == '__main__':

    for i in range(0,24,2):
        usecols = i

        left_data, right_data = get_csv(usecols)
        left_data, right_data = np.array(left_data), np.array(right_data)

        left_peaks, right_peaks = peak_detection(left_data, right_data)
    

    

    
    

    
    
    