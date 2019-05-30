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

train_x1 = pd.read_csv('./dataset/90%_overlap_peak/x1.csv')
train_x1.drop(labels=['Unnamed: 0','seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)

test_list = [x.replace('.csv','') for x in os.listdir('./dataset/test')]

segment = len(test_list)//4

test_x1 = pd.DataFrame(columns=train_x1.columns, dtype=np.float32, index = test_list[0:segment])
test_x2 = pd.DataFrame(columns=train_x1.columns, dtype=np.float32, index = test_list[segment:segment*2])
test_x3 = pd.DataFrame(columns=train_x1.columns, dtype=np.float32, index = test_list[segment*2:segment*3])
test_x4 = pd.DataFrame(columns=train_x1.columns, dtype=np.float32, index = test_list[segment*3:])

def mutli_process1(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    for seg_id in test_x1.index:
        seg = pd.read_csv('./dataset/test/' + seg_id + '.csv')
        create_feature.create_features(seg_id, seg, test_x1, 0, 0)
    test_x1.to_csv('./dataset/90%_overlap_peak/test_x1.csv', index=True)


def mutli_process2(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    for seg_id in test_x2.index:
        seg = pd.read_csv('./dataset/test/' + seg_id + '.csv')
        create_feature.create_features(seg_id, seg, test_x2, 0, 0)
    test_x2.to_csv('./dataset/90%_overlap_peak/test_x2.csv', index=True)

def mutli_process3(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    for seg_id in test_x3.index:
        seg = pd.read_csv('./dataset/test/' + seg_id + '.csv')
        create_feature.create_features(seg_id, seg, test_x3, 0, 0)
    test_x3.to_csv('./dataset/90%_overlap_peak/test_x3.csv', index=True)

def mutli_process4(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    for seg_id in tqdm(test_x4.index):
        seg = pd.read_csv('./dataset/test/' + seg_id + '.csv')
        create_feature.create_features(seg_id, seg, test_x4, 0, 0)
    test_x4.to_csv('./dataset/90%_overlap_peak/test_x4.csv', index=True)

if __name__=='__main__':
    a = time.time()
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    p.apply_async(mutli_process1, args=(1,))
    p.apply_async(mutli_process2, args=(2,))
    p.apply_async(mutli_process3, args=(3,))
    p.apply_async(mutli_process3, args=(4,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    b = time.time()
    print('totol run ', str(b - a))