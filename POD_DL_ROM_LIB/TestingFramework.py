from POD_DL_ROM_LIB import Helper
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import os

try:
    import tensorboard
except ImportError as e:
    TB_MODE = False
else:
    TB_MODE = True
    from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestingFramework(object):
    def __init__(self, params, device=DEVICE, log_folder='./testing_results_local/') -> None:

        self.snapshot_train = params['snapshot_train']
        self.snapshot_test = params['snapshot_test']
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

        self.device = device
        self.logs_folder = log_folder
        self.tensorboard = None

        # We perform an 80-20 split for the training and validation set
        self.num_training_samples = int(0.8 * self.num_samples)

    def testing_data_preparation(self):
        print("Load Snapshot Matrix...")
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

        if self.scaling:
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

            self.snapshot_max, self.snapshot_min = Helper.max_min_componentwise(self.snapshot_train_train,
                                                                                self.num_training_samples,
                                                                                self.num_dimension, self.N)

            self.parameter_max, self.parameter_min = Helper.max_min_componentwise_params(self.parameter_train,
                                                                                         self.num_training_samples,
                                                                                         self.parameter_train.shape[0])

        print("Load Testing Snapshot Matrix and Testing Parameter Matrix...")
        self.snapshot_test_test = np.zeros((self.num_dimension * self.N, self.num_test_samples))
        for i in range(self.num_dimension):
            self.snapshot_test_test[i * self.N:(i + 1) * self.N, :] = np.matmul(
                self.U_transpose[:, i * self.N_h:(i + 1) * self.N_h],
                self.snapshot_test[i * self.N_h:(i + 1) * self.N_h, :])
        if self.scaling:
            Helper.scaling_componentwise(self.snapshot_test_test, self.snapshot_max, self.snapshot_min,
                                         self.num_dimension, self.N)
            Helper.scaling_componentwise_params(self.parameter_test, self.parameter_max, self.parameter_min,
                                                self.parameter_test.shape[0])
        pass

    def testing_pipeline(self):
        # We build our input testing pipeline with the help of dataloader
        # We transpose our data for simplicity purpose
        self.snapshot_test_test = np.transpose(self.snapshot_test_test)
        self.parameter_test = np.transpose(self.parameter_test)

        X = torch.from_numpy(self.snapshot_test_test).float()  # intrinsic coordinates - u_N
        y = torch.from_numpy(self.parameter_test).float()  # params - (mu, t)

        X = torch.reshape(X, shape=[-1, self.num_dimension,
                                    int(np.sqrt(self.N)), int(np.sqrt(self.N))])

        dataset_test = torch.utils.data.TensorDataset(X, y)
        self.testing_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False,
                                                          num_workers=0)
        pass

    def testing_loss(self, output, u_N):
        self.loss = self.omega_h * torch.mean(torch.sum(torch.pow(output - u_N, 2), dim=1))

        pass

    def testing(self):
        start_time = time.time()

        log_folder_trained_model = './trained_models/'
        if not os.path.isdir(log_folder_trained_model):
            print('The trained model is not present in the log folder named trained_models')
            exit()
        else:
            self.model = torch.load(log_folder_trained_model + 'model.pth')

        self.testing_data_preparation()
        self.testing_pipeline()

        self.test_output = np.zeros_like(self.snapshot_test_test)
        testLoss = 0
        nBatches = 0
        self.model.eval()
        # Loop over mini batches of data
        with torch.no_grad():
            for batch_idx, (snapshot_data, parameters) in enumerate(self.testing_loader):
                # Forward pass for the validation data
                self.enc, self.nn, self.dec = self.model(snapshot_data, parameters)
                self.test_output[nBatches * self.batch_size:(nBatches + 1) * self.batch_size, :] = self.dec

                # Calculate the loss function corresponding to the outputs
                self.testing_loss(snapshot_data.view(snapshot_data.size(0), -1), self.dec)

                # Calculate the validation losses
                testLoss += self.loss
                nBatches += 1

        time_evolution_error = np.sqrt(np.mean(np.linalg.norm(self.snapshot_test_test - self.test_output, 2, axis=1) ** 2)) / \
                np.sqrt(np.mean(np.linalg.norm(self.snapshot_test_test, 2, axis=1) ** 2))
        # Display model progress on the current validation set
        print('Testing batch Info...')
        print('Average loss on testing set: {0}'.format(testLoss / nBatches))
        print('Time evolution loss for the testing set: {0}'.format(time_evolution_error))
        print('Took: {0} seconds'.format(time.time() - start_time))

        if self.scaling:
            Helper.inverse_scaling_componentwise(self.test_output, self.snapshot_max, self.snapshot_min,
                                                 self.num_dimension, self.N)
            Helper.inverse_scaling_componentwise_params(self.parameter_test, self.parameter_max, self.parameter_min,
                                                        self.parameter_test.shape[1])
        num_instances = self.snapshot_test.shape[1] // self.num_time_steps
        err = np.zeros((num_instances, 1))
        SnapMat = np.zeros_like(self.snapshot_test)
        self.test_output_transpose = self.test_output.transpose()
        for i in range(num_instances):
            for j in range(self.num_dimension):
                SnapMat[j * self.N_h:(j + 1) * self.N_h, i * self.num_time_steps:(i + 1) * self.num_time_steps] = \
                    np.matmul(self.U[j * self.N_h:(j + 1) * self.N_h, :],
                              self.test_output_transpose[j * self.N:(j + 1) * self.N,
                              i * self.num_time_steps:(i + 1) * self.num_time_steps])
            num = np.sqrt(np.mean(np.linalg.norm(
                self.snapshot_test[:, i * self.num_time_steps:(i + 1) * self.num_time_steps] -
                SnapMat[:, i * self.num_time_steps:(i + 1) * self.num_time_steps], 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(
                self.snapshot_test[:, i * self.num_time_steps:(i + 1) * self.num_time_steps], 2, axis=1) ** 2))
            err[i] = num / den
        print('Relative error indicator: {0}'.format(np.mean(err)))

        X = np.linspace(0, 1000, 1000)
        t = self.parameter_test[0:self.num_time_steps, 1]
        [X_grid, t_grid] = np.meshgrid(X, t)
        X_grid = X_grid.T
        t_grid = t_grid.T
        plt.pcolormesh(X_grid, t_grid, SnapMat)
        # plt.plot(t, self.snapshot_test_test[:, 3])
        # plt.plot(t, self.test_output[:, 3])
        plt.show()

        pass
