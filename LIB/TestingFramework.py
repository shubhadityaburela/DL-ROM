import Utilities
import numpy as np
import torch
import os
import sys
from NetworkModel import ConvAutoEncoderDNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestingFramework(object):
    def __init__(self, params, device=DEVICE) -> None:

        self.time_amplitude_train = params['time_amplitude_train']
        self.time_amplitude_test = params['time_amplitude_test']
        self.parameter_train = params['parameter_train']
        self.parameter_test = params['parameter_test']

        self.batch_size = params['batch_size']
        self.num_early_stop = params['num_early_stop']
        self.scaling = params['scaling']
        self.perform_svd = params['perform_svd']
        self.learning_rate = params['learning_rate']
        self.N_h = params['full_order_model_dimension']
        self.N = params['reduced_order_model_dimension']
        self.n = params['encoded_dimension']
        self.omega_h = params['omega_h']
        self.omega_N = params['omega_N']
        self.typeConv = params['typeConv']
        self.totalModes = params['totalModes']

        self.num_time_steps = int(self.time_amplitude_test.shape[1])
        self.num_samples = int(self.time_amplitude_train.shape[1])
        self.num_parameters_instance = self.num_samples // self.num_time_steps

        self.device = device
        self.time_amplitude_test_output = None

    def testing_data_preparation(self):

        if self.scaling:
            Utilities.scaling_componentwise_params(self.parameter_test, self.parameter_max, self.parameter_min,
                                                   self.parameter_test.shape[0])
        pass

    def testing_pipeline(self):
        # We build our input testing pipeline with the help of dataloader
        # We transpose our wildfire_data for simplicity purpose
        self.parameter_test = np.transpose(self.parameter_test)

        y = torch.from_numpy(self.parameter_test).float()  # params - (mu, t)

        dataset_test = torch.utils.data.TensorDataset(y)
        self.testing_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False,
                                                          num_workers=0)

        pass

    def testing(self, log_folder_trained_model='', testing_method='model_based', model=None, weight_folder=None):

        if testing_method == 'model_based':
            if not os.path.isdir(log_folder_trained_model):
                print('The trained model is not present in the log folder named trained_models')
                exit()
            else:
                self.model = torch.load(log_folder_trained_model + '/trained_models/' + 'model.pth')
        elif testing_method == 'weight_based':
            if not os.path.isdir(log_folder_trained_model):
                print('The trained model is not present in the log folder named trained_results_local')
                exit()
            else:
                # Instantiate the model
                if self.typeConv == '2D':
                    conv_shape = (int(np.sqrt(self.N)), int(np.sqrt(self.N)))
                elif self.typeConv == '1D':
                    conv_shape = self.N
                else:
                    conv_shape = self.N
                self.model = ConvAutoEncoderDNN(encoded_dimension=self.n,
                                                conv_shape=conv_shape,
                                                num_params=self.num_parameters_instance,
                                                typeConv=self.typeConv)
                self.model.load_net_weights(weight_folder)
        else:
            self.model = model

        # Reading the scaling factors for the testing wildfire_data
        scaling = np.load(log_folder_trained_model + '/variables/' + 'scaling.npy', allow_pickle=True)
        self.snapshot_max = scaling[0]
        self.snapshot_min = scaling[1]
        self.delta_max = scaling[2]
        self.delta_min = scaling[3]
        self.parameter_max = scaling[4]
        self.parameter_min = scaling[5]

        self.testing_data_preparation()
        self.testing_pipeline()

        self.test_output = None
        self.model.eval()
        # Loop over mini batches of wildfire_data
        with torch.no_grad():
            for batch_idx, parameters in enumerate(self.testing_loader):
                # Forward pass for the testing wildfire_data
                self.nn, self.dec = self.model.forward_test(parameters[0])
                if batch_idx == 0:
                    self.test_output = np.asarray(self.dec)
                else:
                    self.test_output = np.concatenate((self.test_output, self.dec))

        if self.scaling:
            modes_test_output = self.test_output[:, :self.totalModes]
            Utilities.inverse_scaling_componentwise(modes_test_output, self.snapshot_max, self.snapshot_min)
            self.test_output[:, :self.totalModes] = modes_test_output

            if self.N != self.totalModes:
                delta_test_output = self.test_output[:, self.totalModes:]
                Utilities.inverse_scaling_componentwise(delta_test_output, self.delta_max, self.delta_min)
                self.test_output[:, self.totalModes:] = delta_test_output

            Utilities.inverse_scaling_componentwise_params(self.parameter_test, self.parameter_max, self.parameter_min,
                                                           self.parameter_test.shape[1])

        self.time_amplitude_test_output = self.test_output.transpose()

        pass
