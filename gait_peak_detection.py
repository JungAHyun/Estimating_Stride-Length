from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import csv   

names = {1:'공예은', 2:'이현영', 3:'조은혜',  4:'장승완', 5:'정승민', 6:'호종갑'}
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



def save_csv(left, right, id, main, lp, rp):
    global r_cnt
    global l_cnt

    cnt = 1
    index=0
    f = open('gait_data.csv', 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)

    #왼, 오 개수 같은지 확인
    if len(lp) == len(rp):
        same ='yes'
    else:
        same='no'
    
    
    # main이 왼발 왼발이고 개수가 같거나, main이 오른발이고 개수가 다르면, 오른발 편도의 마지막 저장이 HC-1-2-T0-HC
    # main이 오른발일 경우 왼발 편도 시작 TO-HC-1-2-T0
    for _ in range(len(left)):
        if index == 0 and main == 'right foot':
            wr.writerow([id, f'{"L"}{l_cnt}{"_T"}', left[index]])
            index+=1
        elif cnt % 4 == 1:
            wr.writerow([id, f'{"L"}{l_cnt}{"_H"}', left[index]])
            cnt +=1
            index+=1
        elif cnt % 4 == 2:
            wr.writerow([id, f'{"L"}{l_cnt}{"_1"}', left[index]])
            cnt +=1
            index+=1
        elif cnt % 4 == 3:    
            wr.writerow([id, f'{"L"}{l_cnt}{"_2"}', left[index]])
            cnt +=1
            index+=1
        # elif cnt % 4 == 0:
        #     wr.writerow([id, f'{"L"}{l_cnt}{"_T"}', left[index]])

        #     if (main =='right foot' and same == 'no'and i == len(left)-2) or (main =='left foot' and same == 'yes' and i == len(left)-2):
        #         wr.writerow([id, f'{"L"}{l_cnt}{"_H"}', left[index+1]])
        #         index+=1
           
        #     index+=1
        #     l_cnt+=1
        #     cnt +=1

        elif cnt % 4 == 0:
            if (main =='right foot' and same == 'no'and index == len(left)-2) or (main =='left foot' and same == 'yes' and index == len(left)-2):
                wr.writerow([id, f'{"L"}{l_cnt}{"_T"}', left[index]])
                wr.writerow([id, f'{"L"}{l_cnt}{"_H"}', left[index+1]])
                l_cnt+=1
                cnt +=1
                break
            else:
                wr.writerow([id, f'{"L"}{l_cnt}{"_T"}', left[index]])
                index+=1
                l_cnt+=1
                cnt +=1

    # main이 오른발 왼발이고 개수가 같거나, main이 왼발이고 개수가 다르면, 오른발 편도의 마지막 저장이 HC-1-2-T0-HC
    index=0
    cnt = 1
    for _ in range(len(right)):
        if index == 0 and main == 'left foot':
            wr.writerow([id, f'{"R"}{r_cnt}{"_T"}', right[index]])
            index+=1
        elif cnt % 4 == 1:
            wr.writerow([id, f'{"R"}{r_cnt}{"_H"}', right[index]])
            cnt +=1
            index+=1
        elif cnt % 4 == 2:
            wr.writerow([id, f'{"R"}{r_cnt}{"_1"}', right[index]])
            cnt +=1
            index+=1
        elif cnt % 4 == 3:    
            wr.writerow([id, f'{"R"}{r_cnt}{"_2"}', right[index]])
            cnt +=1
            index+=1
        elif cnt % 4 == 0:
            if (main =='right foot' and same == 'yes' and index == len(right)-2) or (main =='left foot' and same == 'no' and index == len(right)-2):
                wr.writerow([id, f'{"R"}{r_cnt}{"_T"}', right[index]])
                wr.writerow([id, f'{"R"}{r_cnt}{"_H"}', right[index+1]])
                r_cnt+=1
                cnt +=1
                break
            else:
                wr.writerow([id, f'{"R"}{r_cnt}{"_T"}', right[index]])
                index+=1
                r_cnt+=1
                cnt +=1

    wr.writerow(['X', 'X', 'X'])
    f.close()



# peak 추출
def peak_detection(left_data, right_data):
    left_peaks, _ = find_peaks(left_data, distance=25, height=200)
    right_peaks, _ = find_peaks(right_data,  distance=25, height=200)
    
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
    index = 0
    left_toe_off = []
    right_toe_off = []
    toe_off_cnt = 0

    while(index+5<len(left_data)):
        #if left_data[index]<50 and left_data[index+1]<50 and left_data[index+2] and left_data[index+3] < 50 and toe_off_cnt == 0:       #이현영
        if (sum(left_data[index:index+10])== 0) and toe_off_cnt == 0:                     #다른 피험자
            left_toe_off.append(index+1)
            toe_off_cnt = 1
            index+=40
        if left_data[index] > 50 and toe_off_cnt == 1:
            toe_off_cnt = 0
        index+=1
    
    index = 0    
    toe_off_cnt = 0
    while(index+5<len(right_data)):
        #if right_data[index] <50 and right_data[index+1] < 50 and right_data[index+2] < 50 and right_data[index+3] < 50 and toe_off_cnt == 0:                 #이현영
        if (sum(right_data[index:index+10])== 0) and toe_off_cnt == 0:               #다른 피험자
            right_toe_off.append(index+1)
            toe_off_cnt = 1
            index+=40
        if right_data[index] > 50 and toe_off_cnt == 1:
            toe_off_cnt = 0

        index+=1

    # if 31 in left_toe_off:
    #     left_toe_off.remove(31)
    # elif 31 in right_toe_off:
    #     right_toe_off.remove(31)

    return left_toe_off, right_toe_off

# 보행 데이터 정렬 및 통합
def data_integration(lp, rp, lhc, rhc, lto, rto, main):
    cnt = 1
    l_data = []
    r_data = []
    hc_cnt = 0
    p_cnt=0
    to_cnt=0

    print('lp: ', lp, ':: len', len(lp))
    print('rp: ', rp, ':: len', len(rp))
    print('lhc: ', lhc, ':: len', len(lhc))
    print('rhc: ', rhc, ':: len', len(rhc))
    print('lto: ', lto, ':: len', len(lto))
    print('rto: ', rto, ':: len', len(rto))


    #left 데이터 정렬
    while(1):
        try:
            if main == 'right foot' and to_cnt == 0:
                l_data.append(lto[to_cnt])
                to_cnt+=1
            if cnt % 4 == 1:
                l_data.append(lhc[hc_cnt])
                hc_cnt+=1
                cnt +=1
            elif cnt % 4 == 2:
                l_data.append(lp[p_cnt])
                l_data.append(lp[p_cnt+1])
                p_cnt+=2
                cnt +=2
            elif cnt % 4 == 0:
                l_data.append(lto[to_cnt])
                to_cnt+=1
                cnt +=1
        except :
            break


    cnt = 1
    hc_cnt = 0
    p_cnt=0
    to_cnt=0
    while(1):
        try:
            if main == 'left foot'and to_cnt == 0:
                r_data.append(rto[to_cnt])
                to_cnt+=1
            if cnt % 4 == 1:
                r_data.append(rhc[hc_cnt])
                hc_cnt+=1
                cnt +=1
            elif cnt % 4 == 2:
                r_data.append(rp[p_cnt])
                r_data.append(rp[p_cnt+1])
                p_cnt+=2
                cnt +=2
            elif cnt % 4 == 0:
                r_data.append(rto[to_cnt])
                to_cnt+=1
                cnt +=1
        except :
            break



    print('l_data: ',l_data)
    print('r_data: ',r_data)

    
    return l_data, r_data
    
        
# 마지막 남는 peak 제거
# 마지막 peak가 마지막 TO보다 크면 제거
def remove_unnecessary_data(left_peaks, right_peaks,left_toe_off, right_toe_off):
    if left_peaks[-1] >  left_toe_off[-1]:
        left_peaks = np.delete(left_peaks, [-1,-2])
    if right_peaks[-1] >  right_toe_off[-1]:
        right_peaks = np.delete(right_peaks, [-1,-2])
    return left_peaks, right_peaks
        

    
if __name__ == '__main__':

    oneway = 1
    main = 'right foot'
    
    for id in range(1,len(names)+1):
        r_cnt = 1
        l_cnt = 1
        
        name = names[id]
        print('')
        print(id, ': ', name, '-----------------------------------------------------------------------')       
        #편도(one-way) 1,3: 오른발 출발  2,4: 왼발 출발
        for i in range(0,24,2):
            usecols = i

            if oneway == 1 or oneway==3:
                main = 'right foot'
            else:
                main = 'left foot'


            left_data, right_data = get_csv(usecols)
            left_peaks, right_peaks= peak_detection( np.array(left_data),np.array(right_data))
            left_heel_contect, right_heel_contect = heel_contact_detection(left_data, right_data)
            left_toe_off, right_toe_off = toe_off_detection(left_data, right_data)
            
            left_peaks, right_peaks = remove_unnecessary_data(left_peaks, right_peaks,left_toe_off, right_toe_off)

            l_data,r_data = data_integration(left_peaks, right_peaks, left_heel_contect, right_heel_contect, left_toe_off, right_toe_off, main)
            save_csv(l_data, r_data, id, main, left_peaks, right_peaks)        #(left_data, right_data, id, start, end)
            
            
        
            if oneway == 4:
                oneway = 1
            else:
                oneway +=1

            print('-------------------편도 구분선--------------------')

       