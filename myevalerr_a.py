import argparse
import os
import os.path as osp
import sys
import numpy as np
import pickle
import multiprocessing
import importlib
from joblib import Parallel, delayed
from scipy.io import loadmat, savemat
from pathlib import Path

def load_dist(filepath):
    print("Loading load_dist")
    data = loadmat(filepath)
    return np.asarray(data['dist'], dtype=np.float32)

def run():
    id0 = 'horse_01'
    dist = load_dist(osp.join('exp/data/SMAL_r/dist', f'{id0}.mat'))
    print(dist)

if __name__ == '__main__':
    run()