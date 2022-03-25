import numpy as np
import sys
import pathlib
import os
import matplotlib.pyplot as plt

sys.path.append('./../POD_DL_ROM_LIB/')

from TrainingFramework import TrainingFramework
from TestingFramework import TestingFramework
import Helper


def generate():
    # Training and validation data
    SnapMat100 = np.load('SnapShotMatrix100.npy')
    SnapMat200 = np.load('SnapShotMatrix200.npy')
    SnapMat300 = np.load('SnapShotMatrix300.npy')
    SnapMat400 = np.load('SnapShotMatrix400.npy')
    SnapMat500 = np.load('SnapShotMatrix500.npy')
    SnapMat600 = np.load('SnapShotMatrix600.npy')
    SnapMat700 = np.load('SnapShotMatrix700.npy')
    SnapMat800 = np.load('SnapShotMatrix800.npy')
    SnapMat900 = np.load('SnapShotMatrix900.npy')
    Time = np.load('Time.npy')
    # Testing data
    SnapMat540 = np.load('SnapShotMatrix540.npy')
    SnapMat560 = np.load('SnapShotMatrix560.npy')
    SnapMat580 = np.load('SnapShotMatrix580.npy')

    Nx = int(len(SnapMat540) / 2)
    Nt = 500

    SnapMat100 = SnapMat100[0:Nx, 0:Nt]
    SnapMat200 = SnapMat200[0:Nx, 0:Nt]
    SnapMat300 = SnapMat300[0:Nx, 0:Nt]
    SnapMat400 = SnapMat400[0:Nx, 0:Nt]
    SnapMat500 = SnapMat500[0:Nx, 0:Nt]
    SnapMat600 = SnapMat600[0:Nx, 0:Nt]
    SnapMat700 = SnapMat700[0:Nx, 0:Nt]
    SnapMat800 = SnapMat800[0:Nx, 0:Nt]
    SnapMat900 = SnapMat900[0:Nx, 0:Nt]
    Time = Time[0:Nt]

    SnapMat540 = SnapMat540[0:Nx, 0:Nt]
    SnapMat560 = SnapMat560[0:Nx, 0:Nt]
    SnapMat580 = SnapMat580[0:Nx, 0:Nt]

    Param_train = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
    Param_test = np.array([560])

    A = np.concatenate(
        (SnapMat100, SnapMat200, SnapMat300, SnapMat400, SnapMat500, SnapMat600, SnapMat700, SnapMat800, SnapMat900),
        axis=1)
    B = SnapMat560

    C = np.zeros((2, A.shape[1]), dtype=float)
    D = np.zeros((2, B.shape[1]), dtype=float)

    for i in range(len(Param_train)):
        for j in range(len(Time)):
            C[0, i * len(Time) + j] = Param_train[i]
            C[1, i * len(Time) + j] = Time[j]

    for i in range(len(Param_test)):
        for j in range(len(Time)):
            D[0, i * len(Time) + j] = Param_test[i]
            D[1, i * len(Time) + j] = Time[j]

    return A, B, C, D


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Loading training and testing data for both Snapshots and parameters
    custom = False
    if custom:
        TrainingSnapShotMatrix, TestingSnapShotMatrix, TrainingParameterMatrix, TestingParameterMatrix = generate()
    else:
        TrainingSnapShotMatrix = np.load('snapshot_train_frame.npy', allow_pickle=True)
        TestingSnapShotMatrix = np.load('snapshot_test_frame.npy', allow_pickle=True)
        TrainingParameterMatrix = np.load('params_train.npy', allow_pickle=True)
        TestingParameterMatrix = np.load('params_test.npy', allow_pickle=True)

    # Parameters needed for the training and validation of the framework
    params = {
        'FOM': True,  # This switch is true for full order model input and false for only time amplitude matrix
        'snapshot_train': TrainingSnapShotMatrix[0],
        'snapshot_test': TestingSnapShotMatrix[0],
        'time_amplitude_train': None,
        'time_amplitude_test': None,
        'parameter_train': TrainingParameterMatrix[0],
        'parameter_test': TestingParameterMatrix[0],
        'num_parameters': int(TrainingParameterMatrix[0].shape[0]),  # n_mu + 1
        'num_time_steps': 400,  # N_t
        'num_samples': int(TrainingSnapShotMatrix[0].shape[1]),  # N_train x N_t
        'num_test_samples': int(TestingSnapShotMatrix[0].shape[1]),  # N_test x N_t
        'batch_size': 400,
        'num_early_stop': 1500,  # Number of epochs for the early stopping
        'scaling': True,  # true if the data should be scaled
        'perform_svd': 'randomized',  # '', 'normal', 'randomized'
        'learning_rate': 0.001,  # eta  0.001
        'full_order_model_dimension': int(TrainingSnapShotMatrix[0].shape[0]),  # N_h
        'reduced_order_model_dimension': 5,  # N
        'encoded_dimension': 7,  # dimension of the system after the encoder
        'num_dimension': 1,  # Number of dimensions (d)
        'omega_h': 0.8,
        'omega_N': 0.2,
        'typeConv': '1D'
    }

    # POD_DL_ROM = TrainingFramework(params, split=0.67)
    # POD_DL_ROM.training(epochs=50, save_every=10, print_every=10)
    #
    # sys.exit()

    testing_method = 'weight_based'
    if testing_method == 'model_based':
        log_folder_base = 'training_results_local/'
        num_frame_models = 1
        log_folder_trained_model = []
        for num_frame in range(num_frame_models):
            f = sorted(pathlib.Path(log_folder_base).glob('*/'), key=os.path.getmtime)[-(num_frame + 1)]
            log_folder_trained_model.append(f)
        log_folder_trained_model.reverse()

        time_amplitudes_predicted = []
        # Testing for each frame
        for frame, folder_name in enumerate(log_folder_trained_model):
            # Testing for collection of snapshots
            test_model = TestingFramework(params)
            test_model.testing(log_folder_trained_model=str(folder_name), testing_method='model_based')
            time_amplitudes_predicted.append(test_model.time_amplitude_test_output)
    else:
        log_folder_base = 'training_results_local/'
        num_frame_models = 1
        log_folder_trained_model = []
        for num_frame in range(num_frame_models):
            f = sorted(pathlib.Path(log_folder_base).glob('*/'), key=os.path.getmtime)[-(num_frame + 1)]
            log_folder_trained_model.append(f)
        log_folder_trained_model.reverse()

        time_amplitudes_predicted = []
        # Testing for each frame
        for frame, folder_name in enumerate(log_folder_trained_model):
            # Testing for collection of snapshots
            test_model = TestingFramework(params)
            test_model.testing(log_folder_trained_model=str(folder_name), testing_method='weight_based')
            time_amplitudes_predicted.append(test_model.time_amplitude_test_output)

    # Plot and testing
    Nt = params['num_time_steps']
    num_instances = params['snapshot_test'].shape[1] // Nt

    N_h = params['full_order_model_dimension']
    N = params['reduced_order_model_dimension']
    num_dim = params['num_dimension']

    NN_err = np.zeros((num_instances, 1))
    POD_err = np.zeros((num_instances, 1))
    t_err = np.zeros((num_instances, 1))
    SnapMat_NN = np.zeros_like(params['snapshot_test'])
    SnapMat_POD = np.zeros_like(params['snapshot_test'])
    time_amplitudes = np.zeros((num_dim * N, Nt))

    U = Helper.PerformRandomizedSVD(params['snapshot_train'], N, N_h, num_dim)

    for fr in range(num_frame_models):
        for i in range(num_instances):
            for j in range(params['num_dimension']):
                SnapMat_NN[j * N_h:(j + 1) * N_h, i * Nt:(i + 1) * Nt] = \
                    np.matmul(U[j * N_h:(j + 1) * N_h, :],
                              time_amplitudes_predicted[fr][j * N:(j + 1) * N, i * Nt:(i + 1) * Nt])
                time_amplitudes[j * N:(j + 1) * N, :] = np.matmul(
                    U.transpose()[:, j * N_h:(j + 1) * N_h], params['snapshot_test'][j * N_h:(j + 1) * N_h, i * Nt:(i + 1) * Nt])
                SnapMat_POD[j * N_h:(j + 1) * N_h, i * Nt:(i + 1) * Nt] = \
                    np.matmul(U[j * N_h:(j + 1) * N_h, :],
                              time_amplitudes[j * N:(j + 1) * N, :])

            num = np.sqrt(np.mean(np.linalg.norm(
                time_amplitudes[:, :] -
                time_amplitudes_predicted[fr][:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(
                time_amplitudes[:, :], 2, axis=1) ** 2))
            t_err[i] = num / den

            num = np.sqrt(np.mean(np.linalg.norm(
                params['snapshot_test'][:, i * Nt:(i + 1) * Nt] -
                SnapMat_NN[:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(
                params['snapshot_test'][:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            NN_err[i] = num / den

            num = np.sqrt(np.mean(np.linalg.norm(
                params['snapshot_test'][:, i * Nt:(i + 1) * Nt] -
                SnapMat_POD[:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(
                params['snapshot_test'][:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            POD_err[i] = num / den

            print('Relative time amplitude error indicator: {0}'.format(t_err[i]))
            print('Relative NN reconstruction error indicator: {0}'.format(NN_err[i]))
            print('Relative POD reconstruction error indicator: {0}'.format(POD_err[i]))
            print('\n')

            X = np.linspace(0, 400, 400)
            plt.plot(X, time_amplitudes[0, :])
            plt.plot(X, time_amplitudes_predicted[fr][0, i * Nt:(i + 1) * Nt])
            plt.show()

        print(np.mean(t_err), np.mean(NN_err), np.mean(POD_err))

        # X = np.linspace(0, 800, 800)
        # t = params['parameter_test'][4, 0:400]
        # [X_grid, t_grid] = np.meshgrid(X, t)
        # X_grid = X_grid.T
        # t_grid = t_grid.T
        # plt.pcolormesh(X_grid, t_grid, SnapMat_NN)
        # plt.show()


