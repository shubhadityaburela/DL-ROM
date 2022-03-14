import numpy as np
import sys

sys.path.append('./../POD_DL_ROM_LIB/')

from TrainingFramework import TrainingFramework


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
    Param_test = np.array([540])

    A = np.concatenate(
        (SnapMat100, SnapMat200, SnapMat300, SnapMat400, SnapMat500, SnapMat600, SnapMat700, SnapMat800, SnapMat900),
        axis=1)
    B = SnapMat540

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
    custom = True
    if custom:
        TrainingSnapShotMatrix, TestingSnapShotMatrix, TrainingParameterMatrix, TestingParameterMatrix = generate()
    else:
        TrainingSnapShotMatrix = np.load()
        TestingSnapShotMatrix = np.load()
        TrainingParameterMatrix = np.load()
        TestingParameterMatrix = np.load()

    # Parameters needed for the training and validation of the framework
    params = {
        'FOM': True,  # This switch is true for full order model input and false for only time amplitude matrix
        'snapshot_train': TrainingSnapShotMatrix,
        'snapshot_test': TestingSnapShotMatrix,
        'time_amplitude_train': None,
        'time_amplitude_test': None,
        'parameter_train': TrainingParameterMatrix,
        'parameter_test': TestingParameterMatrix,
        'num_parameters': int(TrainingParameterMatrix.shape[0]),  # n_mu + 1
        'num_time_steps': 500,  # N_t
        'num_samples': int(TrainingSnapShotMatrix.shape[1]),  # N_train x N_t
        'num_test_samples': int(TestingSnapShotMatrix.shape[1]),  # N_test x N_t
        'batch_size': 500,
        'num_early_stop': 1500,  # Number of epochs for the early stopping
        'restart': None,  # true if restart is selected
        'scaling': True,  # true if the data should be scaled
        'perform_svd': 'randomized',  # '', 'normal', 'randomized'
        'learning_rate': 0.001,  # eta  0.001
        'full_order_model_dimension': int(TrainingSnapShotMatrix.shape[0]),  # N_h
        'reduced_order_model_dimension': 16,  # N
        'encoded_dimension': 4,  # dimension of the system after the encoder
        'num_dimension': 1,  # Number of channels (d)
        'omega_h': 0.8,
        'omega_N': 0.2,
        'n_h': None,
        'typeConv': None
    }

    POD_DL_ROM = TrainingFramework(params)
    POD_DL_ROM.training(10)

    # TESTING = TestingFramework(params)
    # TESTING.testing()
