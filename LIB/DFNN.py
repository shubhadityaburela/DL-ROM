# pytorch mlp for regression
import numpy as np
import torch.nn
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
import sys
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU, ELU, LeakyReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn import L1Loss, HuberLoss
from torch.nn.init import xavier_uniform_
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

import torch

if torch.cuda.is_available():
    print("The current device: ", torch.cuda.current_device())
    print("Name of the device: ", torch.cuda.get_device_name(0))
    print("Number of GPUs available: ", torch.cuda.device_count())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

import Utilities

import matplotlib.pyplot as plt


def scale_params(PARAMS_TEST, params, scaling):
    if params['scaling']:
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


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, ta_train, p_train, n_outputs):
        # store the inputs and outputs
        self.X = torch.transpose(torch.from_numpy(p_train.astype('float32')), 0, 1)
        self.y = torch.transpose(torch.from_numpy(ta_train.astype('float32')), 0, 1)
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), n_outputs))

        self.X = self.X.to(DEVICE)
        self.y = self.y.to(DEVICE)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.input = Linear(n_inputs, 25)
        xavier_uniform_(self.input.weight)
        self.act1 = ELU()
        # first hidden layer
        self.hidden1 = Linear(25, 50)
        xavier_uniform_(self.hidden1.weight)
        self.act2 = ELU()
        # second hidden layer
        self.hidden2 = Linear(50, 75)
        xavier_uniform_(self.hidden2.weight)
        self.act3 = ELU()
        # third hidden layer
        self.hidden3 = Linear(75, 50)
        xavier_uniform_(self.hidden3.weight)
        self.act4 = LeakyReLU()
        # output layer and output
        self.output = Linear(50, n_outputs)
        xavier_uniform_(self.output.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.input(X)
        X = self.act1(X)
        # first hidden layer
        X = self.hidden1(X)
        X = self.act2(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act3(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act4(X)
        # output layer and output
        X = self.output(X)
        return X


# prepare the dataset
def prepare_data(ta_train, p_train, n_outputs, batch_size):
    # load the dataset
    dataset = CSVDataset(ta_train, p_train, n_outputs)
    # calculate split
    train, val = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


def CombinedLoss(y, y_t, params):
    L1 = L1Loss()
    MSE = MSELoss()

    A = y[:, :params['totalModes']]
    delta = y[:, params['totalModes']:]
    A_t = y_t[:, :params['totalModes']]
    delta_t = y_t[:, params['totalModes']:]

    return L1(delta, delta_t) + L1(A, A_t)


# train the model
def train_model(train_dl, val_dl, n_outputs, model, params, epochs, lr=0.01, loss_type='L1'):
    # define the optimization
    if loss_type == 'L1':
        criterion = L1Loss()
    elif loss_type == 'MSE':
        criterion = MSELoss()
    elif loss_type == 'Huber':
        criterion = HuberLoss(delta=1e-3)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    best_so_far = 1e12  # For early stopping criteria
    trigger_times_early_stopping = 0
    patience_early_stopping = params['num_early_stop']
    # enumerate epochs
    for epoch in range(epochs):
        # enumerate mini batches
        trainLoss = 0
        nBatches = 0
        model.train()
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            if loss_type == 'L1+MSE':
                loss = CombinedLoss(yhat, targets, params)
            else:
                loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

            trainLoss += loss.item()
            nBatches += 1

        model.eval()
        valLoss = 0
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(val_dl):
            # evaluate the model on the validation set
            yhat = model(inputs)
            # calculate validation loss
            if loss_type == 'L1+MSE':
                loss = CombinedLoss(yhat, targets, params)
            else:
                loss = criterion(yhat, targets)
            valLoss += loss.item()
            # retrieve numpy array
            yhat = yhat.cpu().detach().numpy()
            actual = targets.cpu().numpy()
            actual = actual.reshape((len(actual), n_outputs))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)

        num = np.sqrt(np.mean(np.linalg.norm(actuals - predictions, 2, axis=1) ** 2))
        den = np.sqrt(np.mean(np.linalg.norm(actuals, 2, axis=1) ** 2))
        rel_err = num / den

        mse = mean_squared_error(actuals, predictions)

        if valLoss / nBatches > best_so_far:
            trigger_times_early_stopping += 1
            if trigger_times_early_stopping >= patience_early_stopping:
                print('\n')
                print('Early stopping....')
                print('Validation loss did not improve from {} thus exiting the epoch '
                      'loop.........'.format(best_so_far))
                break
        else:
            best_so_far = valLoss / nBatches
            trigger_times_early_stopping = 0

        if epoch % 500 == 0:
            print('Epoch {0}::loss(training):{1:.5f}, loss(validation):{2:.5f}, rel err(validation):{3:.5f}'.
                  format(epoch, trainLoss / nBatches, valLoss / nBatches, rel_err))


def test_model(TA_TEST, params_test, trained_model=None, saved_model=True,
               PATH_TO_WEIGHTS='', params=None, scaling=None, batch_size=None,
               test_shifts_separately=False, num_test_shifts=None, trained_gbrt=None,
               saved_gbrt=False, PATH_TO_GBRT=''):
    if params['scaling']:
        # Reading the scaling factors for the testing wildfire_data
        snapshot_max = scaling[0]
        snapshot_min = scaling[1]
        delta_max = scaling[2]
        delta_min = scaling[3]
        parameter_max = scaling[4]
        parameter_min = scaling[5]

    # test the model
    if test_shifts_separately:
        SHIFTS_TEST = TA_TEST[-num_test_shifts:, :]

        X_test = np.transpose(params_test)
        y_test = np.transpose(SHIFTS_TEST)
        SHIFTS_PRED = np.zeros_like(y_test)
        # Test the model
        if saved_gbrt:
            import joblib
            model_gbrt = joblib.load(PATH_TO_GBRT + 'gbrt.pkl')
        else:
            model_gbrt = trained_gbrt
        for idx in range(len(model_gbrt)):
            y_pred = model_gbrt[idx].predict(X_test)

            SHIFTS_PRED[:, idx] = y_pred

        TA_TEST = TA_TEST[:-num_test_shifts, :]
        n_outputs = np.size(TA_TEST, 0)
    else:
        n_outputs = np.size(TA_TEST, 0)

    test_set = CSVDataset(TA_TEST, params_test, n_outputs)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    numParams = np.size(params_test, 0)

    # define the network
    if saved_model:
        model = MLP(numParams, n_outputs)
        model.load_state_dict(torch.load(PATH_TO_WEIGHTS, map_location=DEVICE))
    else:
        model = trained_model

    model.to(DEVICE)

    model.eval()
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.cpu().detach().numpy()
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), n_outputs))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)

    if test_shifts_separately:
        predictions = np.concatenate((predictions, SHIFTS_PRED), axis=1)
        actuals = np.concatenate((actuals, np.transpose(SHIFTS_TEST)), axis=1)

    if params['scaling']:
        modes_test_output = predictions[:, :params['totalModes']]
        Utilities.inverse_scaling_componentwise(modes_test_output,
                                                snapshot_max, snapshot_min)
        predictions[:, :params['totalModes']] = modes_test_output

        if params['reduced_order_model_dimension'] != params['totalModes']:
            delta_test_output = predictions[:, params['totalModes']:]
            Utilities.inverse_scaling_componentwise(delta_test_output,
                                                    delta_max, delta_min)
            predictions[:, params['totalModes']:] = delta_test_output

    # calculate mse
    mse = mean_squared_error(actuals, predictions)

    num = np.sqrt(np.mean(np.linalg.norm(actuals - predictions, 2, axis=1) ** 2))
    den = np.sqrt(np.mean(np.linalg.norm(actuals, 2, axis=1) ** 2))
    rel_err = num / den

    return rel_err, np.transpose(predictions)


def run_model(TA_TRAIN, params_train, epochs, lr, loss_type,
              logs_folder, pretrained_load=False, pretrained_weights='',
              params=None, batch_size=None, train_shifts_separately=False,
              num_train_shifts=None):
    log_folder = logs_folder + '/' + time.strftime("%Y_%m_%d__%H-%M-%S", time.localtime()) + '/'
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    scaling = 0
    if params['scaling']:
        TA_TRAIN, params_train, scaling = scale_data(TA_TRAIN, params_train, params)

    numParams = np.size(params_train, 0)
    if train_shifts_separately:
        # prepare the data
        SHIFTS_TRAIN = TA_TRAIN[-num_train_shifts:, :]
        X_train, X_val, y_train, y_val = train_test_split(np.transpose(params_train), np.transpose(SHIFTS_TRAIN),
                                                          train_size=0.7)

        model_regressor = []
        for idx in range(num_train_shifts):
            # Best regressor model
            best_regressor = GradientBoostingRegressor(
                max_depth=15,
                n_estimators=20000,
                max_leaf_nodes=4,
                learning_rate=0.1,
                min_samples_split=5,
                loss='absolute_error',
                subsample=0.8
            )
            best_regressor.fit(X_train, y_train[:, idx])

            y_pred = best_regressor.predict(X_val)

            num = np.linalg.norm(y_val[:, idx] - y_pred)
            den = np.linalg.norm(y_val[:, idx])
            rel_err = num / den

            print("evaluation error for shift {} is {}".format(idx, rel_err))

            model_regressor.append(best_regressor)

        TA_TRAIN = TA_TRAIN[:-num_train_shifts, :]
        numOutputs = np.size(TA_TRAIN, 0)
    else:
        model_regressor = []
        numOutputs = np.size(TA_TRAIN, 0)

    # prepare the data
    train_dl, val_dl = prepare_data(TA_TRAIN, params_train, numOutputs, batch_size=batch_size)

    # define the network
    model = MLP(numParams, numOutputs)

    # load pretrained weights
    if pretrained_load:
        model.load_state_dict(torch.load(pretrained_weights, map_location=DEVICE))

    # Move the model to DEVICE
    model.to(DEVICE)

    # train the model
    train_model(train_dl, val_dl, numOutputs, model, params, epochs=epochs, lr=lr, loss_type=loss_type)

    # Save the model
    log_folder_trained_model = log_folder + '/trained_weights/'
    if not os.path.isdir(log_folder_trained_model):
        os.makedirs(log_folder_trained_model)

    torch.save(model.state_dict(), log_folder_trained_model + 'weights.pt')

    import pickle
    with open(log_folder + 'gbrt.pkl', 'wb') as f:
        pickle.dump(model_regressor, f)

    log_folder_variables = log_folder + '/variables/'
    if not os.path.isdir(log_folder_variables):
        os.makedirs(log_folder_variables)
    np.save(log_folder_variables + 'scaling.npy', scaling, allow_pickle=True)

    return model, model_regressor, scaling
