import os
import sys
import signal
import time
import pickle
import certifi

import numpy as np
import numpy.matlib
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import datasets, model_selection, svm, metrics
from sklearn.datasets import fetch_openml
signal.signal(signal.SIGINT, signal.SIG_DFL)


def main():
    datas = list()
    for i in range(50):
        for j in range(50):
            for k in [1000, 10000, 20000, 30000]:
                log_C = -3. + 0.1 * i
                log_gamma = -3. + 0.1 * j
                data_size = k

                try:
                    with open('./svm_data/X_'+str(round(log_C, 1))+'_'+str(round(log_gamma, 1))+'_'+str(data_size)+'.pickle', 'rb') as f:
                        data = pickle.load(f)
                    datas.append(data)
                except FileNotFoundError as e:
                    print(e)

    datas = np.vstack(datas)
    datas = datas[np.argsort(datas[:,3])]

    print(datas[:10,:])
    print(np.shape(datas))

    print( np.array([ np.mean(datas[:,4][datas[:,2] == i]) for i in [1000, 10000, 20000, 30000]]) / 60.)
    # np.set_printoptions(threshold=10000)
    # print(datas[:,4])

    with open('./svm_data/svm_data.pickle', 'wb') as f:
        pickle.dump(datas, f)


if __name__ == '__main__':
    main()
