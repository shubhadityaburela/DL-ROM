from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os
import sys
import time

import Utilities


def scale_params(PARAMS_TEST, scaling):

    # Reading the scaling factors for the testing wildfire_data
    snapshot_max = scaling[0]
    snapshot_min = scaling[1]
    delta_max = scaling[2]
    delta_min = scaling[3]
    parameter_max = scaling[4]
    parameter_min = scaling[5]

    Utilities.scaling_componentwise_params(PARAMS_TEST, parameter_max, parameter_min,
                                           PARAMS_TEST.shape[0])

    return PARAMS_TEST


def scale_data(TA_TRAIN, params_train, params):

    num_samples = int(TA_TRAIN.shape[1])
    snapshot_max, snapshot_min = Utilities.max_min_componentwise(
        TA_TRAIN[:params['totalModes'], :],
        num_samples)
    modes_mat = TA_TRAIN[:params['totalModes'], :]
    Utilities.scaling_componentwise(modes_mat, snapshot_max, snapshot_min)
    TA_TRAIN[:params['totalModes'], :] = modes_mat

    delta_max = 0
    delta_min = 0
    if params['reduced_order_model_dimension'] != params['totalModes']:
        delta_max, delta_min = Utilities.max_min_componentwise(
            TA_TRAIN[params['totalModes']:, :],
            num_samples)
        delta_mat = TA_TRAIN[params['totalModes']:, :]
        Utilities.scaling_componentwise(delta_mat, delta_max, delta_min)
        TA_TRAIN[params['totalModes']:, :] = delta_mat

    parameter_max, parameter_min = Utilities.max_min_componentwise_params(params_train,
                                                                          num_samples,
                                                                          params_train.shape[0])
    Utilities.scaling_componentwise_params(params_train, parameter_max, parameter_min,
                                           params_train.shape[0])

    # Save the scaling factors for testing the network
    scaling = [snapshot_max, snapshot_min, delta_max, delta_min,
               parameter_max, parameter_min]

    return TA_TRAIN, params_train, scaling


def test_model(TA_TEST, params_test, trained_model_list=None, scaling=None, params=None):

    # Reading the scaling factors for the testing wildfire_data
    snapshot_max = scaling[0]
    snapshot_min = scaling[1]
    delta_max = scaling[2]
    delta_min = scaling[3]
    parameter_max = scaling[4]
    parameter_min = scaling[5]

    X_test = np.transpose(params_test)
    y_test = np.transpose(TA_TEST)

    TA_PRED = np.zeros_like(y_test)
    # Test the model
    for idx in range(len(trained_model_list)):
        y_pred = trained_model_list[idx].predict(X_test)

        if idx < params['totalModes']:
            Utilities.inverse_scaling_componentwise(y_pred, snapshot_max, snapshot_min)
        else:
            Utilities.inverse_scaling_componentwise(y_pred, delta_max, delta_min)

        num = np.linalg.norm(y_test[:, idx] - y_pred)
        den = np.linalg.norm(y_test[:, idx])
        rel_err = num / den

        print("test error for {} mode is {}".format(idx, rel_err))

        TA_PRED[:, idx] = y_pred

    return np.transpose(TA_PRED)


def run_model(TA_TRAIN, params_train, params):

    numParams = np.size(params_train, 0)
    numOutputs = np.size(TA_TRAIN, 0)

    TA_TRAIN, params_train, scaling = scale_data(TA_TRAIN, params_train, params)

    # prepare the data
    X_train, X_val, y_train, y_val = train_test_split(np.transpose(params_train), np.transpose(TA_TRAIN))

    regressor_list = []
    for idx in range(numOutputs):
        # Best regressor model
        best_regressor = GradientBoostingRegressor(
            max_depth=6,
            n_estimators=20000,
            learning_rate=0.001,
            loss='absolute_error',
            subsample=0.7
        )
        best_regressor.fit(X_train, y_train[:, idx])

        y_pred = best_regressor.predict(X_val)

        num = np.linalg.norm(y_val[:, idx] - y_pred)
        den = np.linalg.norm(y_val[:, idx])
        rel_err = num / den

        print("evaluation error for {} mode is {}".format(idx, rel_err))

        regressor_list.append(best_regressor)

    return regressor_list, scaling
