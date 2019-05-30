import os
import time
import warnings
import numpy as np
import pandas as pd
from earth_quake.feature import create_feature
from tqdm import tqdm
from multiprocessing import Pool
warnings.filterwarnings("ignore")

os.chdir('C:/Users/mengnan.yi/PycharmProjects/kaggle/earth_quake')
#print(os.listdir("./"))

NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6
NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500
rows = 150000

train = pd.read_csv('./dataset/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

segments = int(np.floor(train.shape[0] / rows))
window = int((segments-1)+segments)
length = window//3

x1 = pd.DataFrame(index=range(length), dtype=np.float32)
y1 = pd.DataFrame(index=range(length), dtype=np.float32)
x2 = pd.DataFrame(index=range(length), dtype=np.float32)
y2 = pd.DataFrame(index=range(length), dtype=np.float32)
x3 = pd.DataFrame(index=range(length+2), dtype=np.float32)
y3 = pd.DataFrame(index=range(length+2), dtype=np.float32)

def mutli_process1(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    for seg_id in tqdm(range(length)):
        start_idx = int(seg_id * rows * 0.5)
        end_idx = int(seg_id * rows * 0.5 + rows)
        seg = train.iloc[start_idx:end_idx]
        create_feature.create_features(seg_id, seg, x1, start_idx, end_idx)
        y1.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
    x1.to_csv('./output/x1.csv', index=True)
    y1.to_csv('./output/y1.csv', index=True)

def mutli_process2(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    for seg_id in range(length):
        start_idx = int(75000*length + seg_id * rows * 0.5)
        end_idx = int(75000*length + seg_id * rows * 0.5 + rows)
        seg = train.iloc[start_idx:end_idx]
        create_feature.create_features(seg_id, seg, x2, start_idx, end_idx)
        y2.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
    x2.to_csv('./output/x2.csv', index=True)
    y2.to_csv('./output/y2.csv', index=True)

def mutli_process3(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    for seg_id in range(length+2):
        start_idx = int(75000*length*2 + seg_id * rows * 0.5)
        end_idx = int(75000*length*2 + seg_id * rows * 0.5 + rows)
        seg = train.iloc[start_idx:end_idx]
        create_feature.create_features(seg_id, seg, x3, start_idx, end_idx)
        y3.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
    x3.to_csv('./output/x3.csv', index=True)
    y3.to_csv('./output/y3.csv', index=True)

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