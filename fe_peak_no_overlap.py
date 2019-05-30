import os
import time
import warnings
import numpy as np
import pandas as pd
from earth_quake.prepare_feature import feature
from tqdm import tqdm
from multiprocessing import Pool
warnings.filterwarnings("ignore")

#os.chdir('C:/Users/ymn84/PycharmProjects/kaggle/earth_quake')
#print(os.listdir("./"))

NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6
NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500
#rows = 150000
step = 150000
"""
test file part
"""

train = pd.read_csv('./dataset/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

#train['diff']=train['time_to_failure'].diff(1)
#scope = pd.Series(train.loc[train['diff']>1.0].index)

# 0       5656574
# 1      50085878
# 2     104677356
# 3     138772453
# 4     187641820
# 5     218652630
# 6     245829585
# 7     307838917
# 8     338276287
# 9     375377848
# 10    419368880
# 11    461811623
# 12    495800225
# 13    528777115
# 14    585568144
# 15    621985673

# range1 = list(scope[0:3])
# range1.insert(0,0)
# range2 = list(scope[2:6])
# range3 = list(scope[5:8])
# range4 = list(scope[7:11])
# range5 = list(scope[10:13])
# range6 = list(scope[12:])

range1 = [0, 5656574, 50085878, 104677356]
range2 = [104677356, 138772453, 187641820, 218652630]
range3 = [218652630, 245829585, 307838917]
range4 = [307838917, 338276287, 375377848, 419368880]
range5 = [419368880, 461811623, 495800225]
range6 = [495800225, 528777115, 585568144, 621985673]

# length1 = 0
# length2 = 0
# length3 = 0
# length4 = 0
# length5 = 0
# length6 = 0

#
# for i in range(len(range1)):
#     count = 0
#     if i == len(range1) - 1:
#         break
#     start = range1[i]
#     end = range1[i + 1]
#     while end - 150000 >= start:
#         start_idx = end - 150000
#         end_idx = end
#         end -= step
#         count += 1
#     length1 += count
#
# for i in range(len(range2)):
#     count = 0
#     if i == len(range2) - 1:
#         break
#     start = range2[i]
#     end = range2[i + 1]
#     while end - 150000 >= start:
#         start_idx = end - 150000
#         end_idx = end
#         end -= step
#         count += 1
#     length2 += count
#
# for i in range(len(range3)):
#     count = 0
#     if i == len(range3) - 1:
#         break
#     start = range3[i]
#     end = range3[i + 1]
#     while end - 150000 >= start:
#         start_idx = end - 150000
#         end_idx = end
#         end -= step
#         count += 1
#     length3 += count
#
# for i in range(len(range4)):
#     count = 0
#     if i == len(range4) - 1:
#         break
#     start = range4[i]
#     end = range4[i + 1]
#     while end - 150000 >= start:
#         start_idx = end - 150000
#         end_idx = end
#         end -= step
#         count += 1
#     length4 += count
#
# for i in range(len(range5)):
#     count = 0
#     if i == len(range5) - 1:
#         break
#     start = range5[i]
#     end = range5[i + 1]
#     while end - 150000 >= start:
#         start_idx = end - 150000
#         end_idx = end
#         end -= step
#         count += 1
#     length5 += count
#
# for i in range(len(range6)):
#     count = 0
#     if i == len(range6) - 1:
#         break
#     start = range6[i]
#     end = range6[i + 1]
#     while end - 150000 >= start:
#         start_idx = end - 150000
#         end_idx = end
#         end -= step
#         count += 1
#     length6 += count

length1 = 696
length2 = 758
length3 = 594
length4 = 742
length5 = 508
length6 = 839

pbar = tqdm(total=length6)

#pbar2 = tqdm(total2=length2)
#pbar3 = tqdm(total3=length3)

# segments = int(np.floor(train.shape[0] / rows))
# window = int((segments-1)*9+segments)
# length = window//3
#
# print ('window is: ', window)
# print ('length is: ', length)

x1 = pd.DataFrame(index=range(length1), dtype=np.float32)
y1 = pd.DataFrame(index=range(length1), dtype=np.float32)
x2 = pd.DataFrame(index=range(length2), dtype=np.float32)
y2 = pd.DataFrame(index=range(length2), dtype=np.float32)
x3 = pd.DataFrame(index=range(length3), dtype=np.float32)
y3 = pd.DataFrame(index=range(length3), dtype=np.float32)
x4 = pd.DataFrame(index=range(length4), dtype=np.float32)
y4 = pd.DataFrame(index=range(length4), dtype=np.float32)
x5 = pd.DataFrame(index=range(length5), dtype=np.float32)
y5 = pd.DataFrame(index=range(length5), dtype=np.float32)
x6 = pd.DataFrame(index=range(length6), dtype=np.float32)
y6 = pd.DataFrame(index=range(length6), dtype=np.float32)

def mutli_process1(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    seg_id1 = 0
    for i in range(len(range1)):
        if i == len(range1) - 1:
            break
        start = range1[i]
        end = range1[i + 1]
        while end - 150000 >= start:
            start_idx = end - 150000
            end_idx = end
            seg = train.iloc[start_idx:end_idx]
            feature.create_features(seg_id1, seg, x1, start_idx, end_idx)
            y1.loc[seg_id1, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= step
            seg_id1 += 1
    x1.to_csv('./dataset/peak_no_overlap/x1.csv', index=True)
    y1.to_csv('./dataset/peak_no_overlap/y1.csv', index=True)

def mutli_process2(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    seg_id2 = 0
    for i in range(len(range2)):
        if i == len(range2) - 1:
            break
        start = range2[i]
        end = range2[i + 1]
        while end - 150000 >= start:
            start_idx = end - 150000
            end_idx = end
            seg = train.iloc[start_idx:end_idx]
            feature.create_features(seg_id2, seg, x2, start_idx, end_idx)
            y2.loc[seg_id2, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= step
            seg_id2 += 1
    x2.to_csv('./dataset/peak_no_overlap/x2.csv', index=True)
    y2.to_csv('./dataset/peak_no_overlap/y2.csv', index=True)

def mutli_process3(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    seg_id3 = 0
    for i in range(len(range3)):
        if i == len(range3) - 1:
            break
        start = range3[i]
        end = range3[i + 1]
        while end - 150000 >= start:
            start_idx = end - 150000
            end_idx = end
            seg = train.iloc[start_idx:end_idx]
            feature.create_features(seg_id3, seg, x3, start_idx, end_idx)
            y3.loc[seg_id3, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= step
            seg_id3 += 1
    x3.to_csv('./dataset/peak_no_overlap/x3.csv', index=True)
    y3.to_csv('./dataset/peak_no_overlap/y3.csv', index=True)

def mutli_process4(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    seg_id4 = 0
    for i in range(len(range4)):
        if i == len(range4) - 1:
            break
        start = range4[i]
        end = range4[i + 1]
        while end - 150000 >= start:
            start_idx = end - 150000
            end_idx = end
            seg = train.iloc[start_idx:end_idx]
            feature.create_features(seg_id4, seg, x4, start_idx, end_idx)
            y4.loc[seg_id4, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= step
            seg_id4 += 1
    x4.to_csv('./dataset/peak_no_overlap/x4.csv', index=True)
    y4.to_csv('./dataset/peak_no_overlap/y4.csv', index=True)

def mutli_process5(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    seg_id5 = 0
    for i in range(len(range5)):
        if i == len(range5) - 1:
            break
        start = range5[i]
        end = range5[i + 1]
        while end - 150000 >= start:
            start_idx = end - 150000
            end_idx = end
            seg = train.iloc[start_idx:end_idx]
            feature.create_features(seg_id5, seg, x5, start_idx, end_idx)
            y5.loc[seg_id5, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= step
            seg_id5 += 1
    x5.to_csv('./dataset/peak_no_overlap/x5.csv', index=True)
    y5.to_csv('./dataset/peak_no_overlap/y5.csv', index=True)

def mutli_process6(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    seg_id6 = 0
    for i in range(len(range6)):
        if i == len(range6) - 1:
            break
        start = range6[i]
        end = range6[i + 1]
        while end - 150000 >= start:
            start_idx = end - 150000
            end_idx = end
            seg = train.iloc[start_idx:end_idx]
            feature.create_features(seg_id6, seg, x6, start_idx, end_idx)
            y6.loc[seg_id6, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= step
            seg_id6 += 1
            pbar.update(1)
    x6.to_csv('./dataset/peak_no_overlap/x6.csv', index=True)
    y6.to_csv('./dataset/peak_no_overlap/y6.csv', index=True)

if __name__=='__main__':
    a = time.time()
    print('Parent process %s.' % os.getpid())
    p = Pool(6)
    p.apply_async(mutli_process1, args=(1,))
    p.apply_async(mutli_process2, args=(2,))
    p.apply_async(mutli_process3, args=(3,))
    p.apply_async(mutli_process4, args=(4,))
    p.apply_async(mutli_process5, args=(5,))
    p.apply_async(mutli_process6, args=(6,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    b = time.time()
    print('totol run ', str(b - a))