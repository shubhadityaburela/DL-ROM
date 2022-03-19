import Helper
import numpy as np
import torch
import os
from NetworkModel import ConvAutoEncoderDNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestingFramework(object):
    def __init__(self, params, device=DEVICE) -> None:

        self.FOM = params['FOM']
        self.snapshot_train = params['snapshot_train']
        self.snapshot_test = params['snapshot_test']
        self.time_amplitude_train = params['time_amplitude_train']
        self.time_amplitude_test = params['time_amplitude_test']
        self.parameter_train = params['parameter_train']
        self.parameter_test = params['parameter_test']
        self.num_parameters = params['num_parameters']
        self.num_time_steps = params['num_time_steps']
        self.num_samples = params['num_samples']
        self.num_test_samples = params['num_test_samples']
        self.batch_size = params['batch_size']
        self.num_early_stop = params['num_early_stop']
        self.restart = params['restart']
        self.scaling = params['scaling']
        self.perform_svd = params['perform_svd']
        self.learning_rate = params['learning_rate']
        self.N_h = params['full_order_model_dimension']
        self.N = params['reduced_order_model_dimension']
        self.n = params['encoded_dimension']
        self.num_dimension = params['num_dimension']
        self.omega_h = params['omega_h']
        self.omega_N = params['omega_N']
        self.n_h = params['n_h']
        self.typeConv = params['typeConv']

        self.device = device
        self.time_amplitude_test_output = None

        # We perform an 80-20 split for the training and validation set
        self.num_training_samples = int(0.8 * self.num_samples)

    def testing_data_preparation(self):
        if self.FOM:
            # First we need to perform the SVD of the snapshot matrix to create the basis matrix 'self.U'
            # Full_dimension = N_h x N_s, where N_s = N_train x N_t (N_t : time instants, N_train : parameter instances)
            # Reduced dimension = N (the final basis matrix U will be of size N_h x N)
            if self.perform_svd == 'normal':
                self.U = Helper.PerformSVD(self.snapshot_train, self.N, self.N_h, self.num_dimension)
            elif self.perform_svd == 'randomized':
                self.U = Helper.PerformRandomizedSVD(self.snapshot_train, self.N, self.N_h, self.num_dimension)
            else:
                print('Please select of the two options : normal or randomized')
                exit()
            self.U_transpose = np.transpose(self.U)  # making it (U^T)

            # We now perform random permutation of the columns of the 'self.snapshot_train'
            # to better generalize the split
            perm_id = np.random.RandomState(seed=42).permutation(self.snapshot_train.shape[1])
            self.snapshot_train = self.snapshot_train[:, perm_id]
            self.parameter_train = self.parameter_train[:, perm_id]

            # Split the 'self.snapshot_train' matrix into -> 'self.snapshot_train_train'
            self.snapshot_train_train = np.zeros((self.num_dimension * self.N, self.num_training_samples))

            # Compute the intrinsic coordinates for 'self.snapshot_train_train' by performing a projection onto the
            # reduced basis.
            # self.snapshot_train_train = (self.U)^T x self.snapshot_train
            for i in range(self.num_dimension):
                self.snapshot_train_train[i * self.N:(i + 1) * self.N, :] = np.matmul(
                    self.U_transpose[:, i * self.N_h:(i + 1) * self.N_h],
                    self.snapshot_train[i * self.N_h:(i + 1) * self.N_h, :self.num_training_samples])
        else:
            perm_id = np.random.RandomState(seed=42).permutation(self.time_amplitude_train.shape[1])
            self.time_amplitude_train = self.time_amplitude_train[:, perm_id]
            self.parameter_train = self.parameter_train[:, perm_id]

            self.snapshot_train_train = np.zeros((self.num_dimension * self.N, self.num_training_samples))
            self.snapshot_train_train = self.time_amplitude_train[:, :self.num_training_samples]

        if self.scaling:
            self.snapshot_max, self.snapshot_min = Helper.max_min_componentwise(self.snapshot_train_train,
                                                                                self.num_training_samples,
                                                                                self.num_dimension, self.N)
            self.parameter_max, self.parameter_min = Helper.max_min_componentwise_params(self.parameter_train,
                                                                                         self.num_training_samples,
                                                                                         self.parameter_train.shape[0])
            Helper.scaling_componentwise_params(self.parameter_test, self.parameter_max, self.parameter_min,
                                                self.parameter_test.shape[0])

        pass

    def testing_pipeline(self):
        # We build our input testing pipeline with the help of dataloader
        # We transpose our data for simplicity purpose
        self.parameter_test = np.transpose(self.parameter_test)

        y = torch.from_numpy(self.parameter_test).float()  # params - (mu, t)

        dataset_test = torch.utils.data.TensorDataset(y)
        self.testing_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False,
                                                          num_workers=0)

        pass

    def testing(self, log_folder_trained_model='', testing_method='model_based'):

        if testing_method == 'model_based':
            if not os.path.isdir(log_folder_trained_model):
                print('The trained model is not present in the log folder named trained_models')
                exit()
            else:
                self.model = torch.load(log_folder_trained_model + '/trained_models/' + 'model.pth')
        else:
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
                self.model = ConvAutoEncoderDNN(conv_shape=conv_shape, num_params=self.num_parameters,
                                                typeConv=self.typeConv)
                self.model.load_net_weights(log_folder_trained_model + '/net_weights/' + 'best_results.pt')

        self.testing_data_preparation()
        self.testing_pipeline()

        self.test_output = None
        self.model.eval()
        # Loop over mini batches of data
        with torch.no_grad():
            for batch_idx, parameters in enumerate(self.testing_loader):
                # Forward pass for the testing data
                self.nn, self.dec = self.model.forward_test(parameters[0])
                if batch_idx == 0:
                    self.test_output = np.asarray(self.dec)
                else:
                    self.test_output = np.concatenate((self.test_output, self.dec))

        if self.scaling:
            Helper.inverse_scaling_componentwise(self.test_output, self.snapshot_max, self.snapshot_min,
                                                 self.num_dimension, self.N)
            Helper.inverse_scaling_componentwise_params(self.parameter_test, self.parameter_max, self.parameter_min,
                                                        self.parameter_test.shape[1])

        self.time_amplitude_test_output = self.test_output.transpose()

        pass
