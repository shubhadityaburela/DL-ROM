import numpy as np
import sys
import pathlib
import os
import matplotlib.pyplot as plt

sys.path.append('./LIB/')

impath = "./plots/"
os.makedirs(impath, exist_ok=True)

from CADFNN_training import TrainingFramework
from CADFNN_testing import TestingFramework
import Utilities


def create_data(nmodes, mu_vecs, t):
    Nt = len(t)
    Nsamples = np.size(mu_vecs)

    amplitudes_1 = np.zeros([nmodes, Nsamples * Nt])
    amplitudes_2 = np.zeros([nmodes, Nsamples * Nt])
    for i in range(nmodes):
        amplitudes_1[i, :] = np.concatenate(
            np.asarray([(1 + np.sin(mu * t) * np.exp(-mu * t)) * mu * (i + 1) for mu in mu_vecs]))
        amplitudes_2[i, :] = np.concatenate(
            np.asarray([(1 - np.sin(mu * t) * np.exp(-mu * t)) * mu * (i + 1) for mu in mu_vecs]))

    amplitudes = np.concatenate((amplitudes_1, amplitudes_2), axis=0)

    p = [np.squeeze(np.asarray([[t], [np.ones_like(t) * mu]])) for mu in mu_vecs]
    p = np.concatenate(p, axis=1)

    return amplitudes, p


def plot_timeamplitudesTraining(amplitudes_train, nmodes, t):
    Nt = len(t)
    fig, axs = plt.subplots(1, nmodes, sharey=True, figsize=(18, 5), num=1)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, amplitudes_train[k, :Nt], color="red", marker=".", label='frame 1')
        ax.plot(t, amplitudes_train[k + nmodes, :Nt], color="blue", marker=".", label='frame 2')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.grid()
        ax.legend(loc='upper right')
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    fig.savefig(impath + "time_amplitudes_" + "training" + '.png', dpi=600, transparent=True)


def plot_timeamplitudesPredicted(ta_pred, ta_test, nmodes, t):

    time_amplitudes_1_pred = ta_pred[:nmodes, :]
    time_amplitudes_2_pred = ta_pred[nmodes:2 * nmodes, :]

    time_amplitudes_1_test = ta_test[:nmodes, :]
    time_amplitudes_2_test = ta_test[nmodes:2 * nmodes, :]

    # Frame 1
    fig, axs = plt.subplots(1, nmodes, sharey=True, figsize=(18, 5), num=3)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, time_amplitudes_1_pred[k, :], color="red", marker=".", label='predicted')
        ax.plot(t, time_amplitudes_1_test[k, :], color="blue", marker=".", label='actual')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.legend(loc='upper right')
        ax.grid()
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    fig.savefig(impath + "time_amplitudes_frame_1_" + "predicted" + '.png', dpi=600, transparent=True)

    # Frame 2
    fig, axs = plt.subplots(1, nmodes, sharey=True, figsize=(18, 5), num=4)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, time_amplitudes_2_pred[k, :], color="red", marker=".", label='predicted')
        ax.plot(t, time_amplitudes_2_test[k, :], color="blue", marker=".", label='actual')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.legend(loc='upper right')
        ax.grid()
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    fig.savefig(impath + "time_amplitudes_frame_2_" + "predicted" + '.png', dpi=600, transparent=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # DATA PREPARATION  #--------------------------------#
    plot_data = True
    Nx = 500
    Nt = 500  # numer of time intervals
    L = 1  # total domain size
    nmodes = 5  # reduction of singular values
    t = np.linspace(0, np.pi, Nt)
    x = np.arange(-Nx // 2, Nx // 2) / Nx * L
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # %% Create training data
    mu_vecs_train = np.asarray([4, 5, 6])
    Nsamples_train = np.size(mu_vecs_train)
    TA_TRAIN, PARAMS_TRAIN = create_data(nmodes, mu_vecs_train, t)

    # %% Create testing data
    mu_vecs_test = np.asarray([5.5])
    Nsamples_test = np.size(mu_vecs_test)
    TA_TEST, PARAMS_TEST = create_data(nmodes, mu_vecs_test, t)

    # Plot the training data
    plot_timeamplitudesTraining(TA_TRAIN, nmodes, t)

    # NETWORK TRAINING PREPARATION  #--------------------------------#
    # Parameters needed for the training and validation of the framework
    dict_network = {
        'time_amplitude_train': TA_TRAIN,
        'time_amplitude_test': TA_TEST,
        'parameter_train': PARAMS_TRAIN,
        'parameter_test': PARAMS_TEST,
        'batch_size': 100,
        'num_early_stop': 10000,  # Number of epochs for the early stopping
        'pretrained_load': False,  # Whether to initialize the network with pretrained weights
        'scaling': True,  # true if the data should be scaled
        'perform_svd': 'randomized',  # 'normal', 'randomized'
        'learning_rate': 0.05,  # eta
        'full_order_model_dimension': None,  # N_h
        'reduced_order_model_dimension': nmodes * 2,  # N
        'encoded_dimension': 4,  # dimension of the system after the encoder
        'omega_h': 0.8,
        'omega_N': 0.2,
        'typeConv': '1D',  # Type of convolutional layer for the network : '1D' or '2D'
        'totalModes': 2 * nmodes,  # Total number of modes for all the frames
    }

    # Training call
    train_model = TrainingFramework(dict_network, split=0.70, log_folder='./training_results/')
    trained_model = train_model.training(epochs=100, save_every=50, print_every=50, log_base_name='/',
                                         pretrained_weights=None)

    # Testing call
    testing_method = None
    log_folder_base = 'training_results/'
    log_folder_trained_model = sorted(pathlib.Path(log_folder_base).glob('*/'), key=os.path.getmtime)[-1]
    test_model = TestingFramework(dict_network)
    test_model.testing(log_folder_trained_model=str(log_folder_trained_model),
                       testing_method=testing_method, model=trained_model)
    TA_PRED = test_model.time_amplitude_test_output

    # Plot the predicted results
    plot_timeamplitudesPredicted(TA_PRED, TA_TEST, nmodes, t)


