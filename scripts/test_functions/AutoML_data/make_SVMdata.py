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


def main(log_C, log_gamma, data_size):
    # print(certifi.where())
    os.environ['SSL_CERT_FILE'] = certifi.where()

    start_time = time.time()
    mnist_data, mnist_label = fetch_openml('mnist_784', return_X_y=True, data_home='../data')

    # normalize 8 bit data
    mnist_data = mnist_data / 255

    # print(mnist_data.shape)
    # print(mnist_label.shape)
    # (70000, 28 \times 28)
    # (70000,)

    clf = svm.SVC(C=10**log_C, gamma=10**log_gamma)
    score = model_selection.cross_val_score(clf, mnist_data.iloc[:data_size,:], mnist_label.iloc[:data_size], cv=5, n_jobs=5)

    elapsed_time = time.time() - start_time

    with open('./svm_data/X_'+str(round(log_C, 1))+'_'+str(round(log_gamma, 1))+'_'+str(data_size)+'.pickle', 'wb') as f:
        pickle.dump([round(log_C, 1), round(log_gamma, 1), data_size, np.mean(score), elapsed_time], f)

    print([round(log_C, 1), round(log_gamma, 1), data_size, np.mean(score), elapsed_time])


if __name__ == '__main__':
    args = sys.argv
    log_C = -3. + 0.1 * np.float(args[1])
    log_gamma = -3. + 0.1 * np.float(args[2])
    data_size = np.int(args[3])
    main(log_C, log_gamma, data_size)
