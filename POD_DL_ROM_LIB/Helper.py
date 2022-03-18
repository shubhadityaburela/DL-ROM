import os

import numpy as np
import torch.cuda
from scipy import linalg
from sklearn.utils import extmath
from shutil import copy2 as copy_file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def padding(Mat, n):
    # Implements a padding (useful for convolutional autoencoder)
    paddings = np.zeros((Mat.shape[0], n))
    S = np.hstack((Mat, paddings))

    return S


def PerformSVD(Mat, N, N_h, num_dim):
    # print('Computing exact POD...')
    U = np.zeros((num_dim * N_h, N))
    # start_time = time.time()
    for i in range(num_dim):
        U, Sigma, Vh = linalg.svd(Mat[i * N_h:(i + 1) * N_h, :], full_matrices=False,
                                  overwrite_a=False, check_finite=False, lapack_driver='gesvd')
    # print('Done... Took: {0} seconds'.format(time.time() - start_time))

    return U[:, :N]


def PerformRandomizedSVD(Mat, N, N_h, num_dim):
    # print('Computing randomized POD...')
    U = np.zeros((num_dim * N_h, N))
    # start_time = time.time()
    for i in range(num_dim):
        U[i * N_h:(i + 1) * N_h], Sigma, Vh = extmath.randomized_svd(Mat[i * N_h:(i + 1) * N_h, :],
                                                                     n_components=N, transpose=False,
                                                                     flip_sign=False, random_state=123)
    # print('Done: computed matrix V of shape {0}... Took: {1} seconds'.format(U.shape, time.time() - start_time))

    return U


def max_min(Mat, n_train):
    Mat_max = np.max(np.max(Mat[:, :n_train], axis=0), axis=0)  # np.max(Mat, axis = 0) -> max of each column
    Mat_min = np.min(np.min(Mat[:, :n_train], axis=0), axis=0)

    return Mat_max, Mat_min


def scaling(Mat, Mat_max, Mat_min):
    Mat[:] = (Mat - Mat_min) / (Mat_max - Mat_min)


def inverse_scaling(Mat, Mat_max, Mat_min):
    Mat[:] = (Mat_max - Mat_min) * Mat + Mat_min


def max_min_componentwise(Mat, n_train, num_dim, N):
    Mat_max, Mat_min = np.zeros((num_dim, 1)), np.zeros((num_dim, 1))

    for i in range(num_dim):
        Mat_max[i], Mat_min[i] = max_min(Mat[i * N: (i + 1) * N, :], n_train)

    return Mat_max, Mat_min


def scaling_componentwise(Mat, Mat_max, Mat_min, num_dim, N):
    for i in range(num_dim):
        scaling(Mat[i * N: (i + 1) * N, :], Mat_max[i], Mat_min[i])


def inverse_scaling_componentwise(Mat, Mat_max, Mat_min, num_dim, N):
    for i in range(num_dim):
        inverse_scaling(Mat[:, i * N: (i + 1) * N], Mat_max[i], Mat_min[i])
###########################################################


def max_min_componentwise_params(Mat, n_train, num_param):
    Mat_max, Mat_min = np.zeros((num_param, 1)), np.zeros((num_param, 1))

    for i in range(num_param):
        Mat_max[i], Mat_min[i] = max_min(Mat[i, :][np.newaxis], n_train)

    return Mat_max, Mat_min


def scaling_componentwise_params(Mat, Mat_max, Mat_min, n_components):
    for i in range(n_components):
        scaling(Mat[i, :], Mat_max[i], Mat_min[i])


def inverse_scaling_componentwise_params(Mat, Mat_max, Mat_min, n_components):
    for i in range(n_components):
        inverse_scaling(Mat[:, i][np.newaxis], Mat_max[i], Mat_min[i])


def save_codeBase(source_path, dest_path):
    source_path = source_path if source_path.endswith('/') else source_path + '/'
    dest_path = dest_path + 'code/' if dest_path.endswith('/') else dest_path + '/code/'
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
    for fname in os.listdir(source_path):
        if fname.endswith('.py'):
            copy_file(source_path + fname, dest_path + fname)


def to_torch(data, device=DEVICE):
    if (type(data) == list) or (type(data) == tuple):
        data = [to_torch(el, device) for el in data]
    elif type(data) == dict:
        data = {key: to_torch(data[key], device) for key in data}
    else:
        if type(data) != torch.Tensor:
            data = torch.tensor(data, device=device, dtype=torch.float32)
        else:
            data = data.to(device)

    return data

