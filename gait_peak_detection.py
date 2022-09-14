from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import csv   

name = '정승민'
id = 10

path = '(노이즈제거본)Stride_length_experiment_stride_count_feature_extraction.xlsx'


r_cnt = 1
l_cnt = 1

def get_csv(usecols):
    df = pd.read_excel(path, sheet_name=name , usecols = [usecols, usecols+1], skiprows = 1)  #앞의 2행 생략
    left_data = []
    right_data = []
    for i in range(len(df.values)):
        left_data.append(df.values[i][0])
        right_data.append(df.values[i][1])
    
    return left_data, right_data



def save_csv(left, right, id):
    global r_cnt, l_cnt, cnt 
    cnt = 1
    f = open('gait_data.csv', 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)

    for data in left:
            if cnt % 4 == 1:
                wr.writerow([id, f'{"L"}{l_cnt}{"_H"}', data])
                cnt +=1
            elif cnt % 4 == 2:
                wr.writerow([id, f'{"L"}{l_cnt}{"_1"}', data])
                cnt +=1
            elif cnt % 4 == 3:    
                wr.writerow([id, f'{"L"}{l_cnt}{"_2"}', data])
                cnt +=1
            elif cnt % 4 == 0:
                wr.writerow([id, f'{"L"}{l_cnt}{"_T"}', data])
                l_cnt+=1
                cnt +=1

    cnt = 1
    for data in right:
        if cnt % 4 == 1:
            wr.writerow([id, f'{"R"}{r_cnt}{"_H"}', data])
            cnt +=1
        elif cnt % 4 == 2:
            wr.writerow([id, f'{"R"}{r_cnt}{"_1"}', data])
            cnt +=1
        elif cnt % 4 == 3:    
            wr.writerow([id, f'{"R"}{r_cnt}{"_2"}', data])
            cnt +=1
        elif cnt % 4 == 0:
            wr.writerow([id, f'{"R"}{r_cnt}{"_T"}', data])
            r_cnt+=1
            cnt +=1
        
    wr.writerow(['X', 'X', 'X'])

    f.close()


# peak 추출
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


# heel_contact 추출
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
        

# toe_off 추출
def toe_off_detection(left_data, right_data):
    index = 30
    left_toe_off = []
    right_toe_off = []
    toe_off_cnt = 0

    while(index+5<len(left_data)):
        #if left_data[index]<50 and left_data[index+1]<50 and left_data[index+2] and left_data[index+3] < 50 and toe_off_cnt == 0:       #이현영
        if (left_data[index]+left_data[index+1]+left_data[index+2]+left_data[index+3] == 0) and toe_off_cnt == 0:                     #다른 피험자
            left_toe_off.append(index+1)
            toe_off_cnt = 1
            index+=40
        if left_data[index] > 50 and toe_off_cnt == 1:
            toe_off_cnt = 0

        index+=1
    
    index = 30    
    toe_off_cnt = 0
    while(index+5<len(right_data)):
        #if right_data[index] <50 and right_data[index+1] < 50 and right_data[index+2] < 50 and right_data[index+3] < 50 and toe_off_cnt == 0:                 #이현영
        if (right_data[index]+right_data[index+1]+right_data[index+2]+right_data[index+3] == 0) and toe_off_cnt == 0:               #다른 피험자
            right_toe_off.append(index+1)
            toe_off_cnt = 1
            index+=40
        if right_data[index] > 50 and toe_off_cnt == 1:
            toe_off_cnt = 0

        index+=1
    
    if 31 in left_toe_off:
        left_toe_off.remove(31)
    elif 31 in right_toe_off:
        right_toe_off.remove(31)

    return left_toe_off, right_toe_off


# 보행 데이터 정렬 및 통합
def data_integration(lp, rp, lhc, rhc, lto, rto):

    cnt = 1
    l_data = []
    r_data = []

    hc_cnt = 0
    p_cnt=0
    to_cnt=0

    print('lp: ', lp)
    print('rp: ', rp)
    print('lhc: ', lhc)
    print('rhc: ', rhc)
    print('lto: ', lto)
    print('rto: ', rto)


    while(cnt < max(len(lp)/2, len(rp)/2) +max(len(lto),len(rto) )+max(len(lhc),len(rhc))+2):
        # try:
            if cnt % 4 == 1:
                l_data.append(lhc[hc_cnt])
                r_data.append(rhc[hc_cnt])
                hc_cnt+=1
                cnt +=1

            elif cnt % 4 == 2:
                l_data.append(lp[p_cnt])
                l_data.append(lp[p_cnt+1])
                r_data.append(rp[p_cnt])
                r_data.append(rp[p_cnt+1])
                p_cnt+=2
                cnt +=2

            elif cnt % 4 == 0:
                l_data.append(lto[to_cnt])
                r_data.append(rto[to_cnt])
                to_cnt+=1
                cnt +=1
        
        # except IndexError:
            # break


    print('l_data: ',l_data)
    print('r_data: ',r_data)

    return l_data, r_data

    

        
    



if __name__ == '__main__':



    for i in range(0,24,2):
        usecols = i

        left_data, right_data = get_csv(usecols)

        left_peaks, right_peaks= peak_detection( np.array(left_data),np.array(right_data))
        left_heel_contect, right_heel_contect = heel_contact_detection(left_data, right_data)
        left_toe_off, right_toe_off = toe_off_detection(left_data, right_data)
        

        l_data,r_data = data_integration(left_peaks, right_peaks, left_heel_contect, right_heel_contect, left_toe_off, right_toe_off)
        save_csv(l_data, r_data, id)        #(left_data, right_data, id, start, end)
    

    
    

    
    
    