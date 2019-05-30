import os
import time
import warnings
import numpy as np
import pandas as pd
from earth_quake.feature import create_feature
from tqdm import tqdm
from multiprocessing import Pool
warnings.filterwarnings("ignore")

#os.chdir('C:/Users/mengnan.yi/PycharmProjects/kaggle/earth_quake')
#print(os.listdir("./"))

NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6
NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500
rows = 150000

"""
test file part
"""

train = pd.read_csv('./dataset/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

#train['diff']=train['time_to_failure'].diff(1)
#scope = pd.Series(train.loc[train['diff']>1.0].index)
#range1 = list(scope[0:6])
#range1.insert(0,0)
#range2 = list(scope[5:11])
#range3 = list(scope[10:])

range1 = [0, 5656574, 50085878, 104677356, 138772453, 187641820, 218652630]
range2 = [218652630, 245829585, 307838917, 338276287, 375377848, 419368880]
range3 = [419368880, 461811623, 495800225, 528777115, 585568144, 621985673]

# length1 = 0
# length2 = 0
# length3 = 0
# for i in range(len(range1)):
#     count = 0
#     if i == len(range1) - 1:
#         break
#     start = range1[i]
#     end = range1[i + 1]
#     while end - 150000 >= start:
#         start_idx = end - 150000
#         end_idx = end
#         end -= 15000
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
#         end -= 15000
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
#         end -= 15000
#         count += 1
#     length3 += count

length1 = 14520
length2 = 13333
length3 = 13460

pbar = tqdm(total=length1)

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
            create_feature.create_features(seg_id1, seg, x1, start_idx, end_idx)
            y1.loc[seg_id1, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= 15000
            seg_id1 += 1
            pbar.update(1)
    x1.to_csv('./dataset/90%_overlap_peak_new_fe/x1.csv', index=True)
    y1.to_csv('./dataset/90%_overlap_peak_new_fe/y1.csv', index=True)

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
            create_feature.create_features(seg_id2, seg, x2, start_idx, end_idx)
            y2.loc[seg_id2, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= 15000
            seg_id2 += 1
    x2.to_csv('./dataset/90%_overlap_peak_new_fe/x2.csv', index=True)
    y2.to_csv('./dataset/90%_overlap_peak_new_fe/y2.csv', index=True)

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
            create_feature.create_features(seg_id3, seg, x3, start_idx, end_idx)
            y3.loc[seg_id3, 'time_to_failure'] = seg['time_to_failure'].values[-1]
            end -= 15000
            seg_id3 += 1
    x3.to_csv('./dataset/90%_overlap_peak_new_fe/x3.csv', index=True)
    y3.to_csv('./dataset/90%_overlap_peak_new_fe/y3.csv', index=True)

if __name__=='__main__':
    a = time.time()
    print('Parent process %s.' % os.getpid())
    p = Pool(3)
    p.apply_async(mutli_process1, args=(1,))
    p.apply_async(mutli_process2, args=(2,))
    p.apply_async(mutli_process3, args=(3,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    b = time.time()
    print('totol run ', str(b - a))