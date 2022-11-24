import Utilities
import numpy as np
import time
import torch
import os
import sys
import matplotlib.pyplot as plt
from NetworkModel import ConvAutoEncoderDNN

try:
    import tensorboard
except ImportError as e:
    TB_MODE = False
else:
    TB_MODE = True
    from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingFramework(object):
    def __init__(self, params, split=0.8, device=DEVICE, log_folder='./training_results_local/') -> None:

        self.time_amplitude_train = params['time_amplitude_train']
        self.parameter_train = params['parameter_train']

        self.batch_size = params['batch_size']
        self.num_early_stop = params['num_early_stop']
        self.pretrained_load = params['pretrained_load']
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

        self.num_samples = int(self.time_amplitude_train.shape[1])
        self.num_parameters = int(self.parameter_train.shape[0])

        self.device = device
        self.logs_folder = log_folder
        self.tensorboard = None

        self.data_train_val = None
        self.data_train_train = None
        self.parameter_train_val = None
        self.parameter_train_train = None

        self.snapshot_max = None
        self.snapshot_min = None
        self.delta_max = None
        self.delta_min = None
        self.parameter_max = None
        self.parameter_min = None

        self.U = None
        self.U_transpose = None

        self.dec = None
        self.nn = None
        self.enc = None
        self.opt = None

        self.loss = None
        self.loss_N = None
        self.loss_h = None
        self.best_so_far = None

        self.validation_loader = None
        self.training_loader = None
        self.testing_loader = None

        self.model = None

        # We perform an 67-33 split for the training and validation set
        self.num_training_samples = int(split * self.num_samples)
        self.permute = False

    def data_preparation(self):
        print('DATA PREPARATION START...\n')

        self.data_train_train = np.zeros((self.N, self.num_training_samples))
        self.data_train_val = np.zeros((self.N, self.num_samples - self.num_training_samples))

        if self.permute:
            perm_id = np.random.RandomState(seed=42).permutation(self.time_amplitude_train.shape[1])
            self.time_amplitude_train = self.time_amplitude_train[:, perm_id]
            self.parameter_train = self.parameter_train[:, perm_id]

        # Normalize the data
        if self.scaling:
            self.snapshot_max, self.snapshot_min = Utilities.max_min_componentwise(
                self.time_amplitude_train[:self.totalModes, :],
                self.num_samples)
            modes_mat = self.time_amplitude_train[:self.totalModes, :]
            Utilities.scaling_componentwise(modes_mat, self.snapshot_max, self.snapshot_min)
            self.time_amplitude_train[:self.totalModes, :] = modes_mat

            if self.N != self.totalModes:
                self.delta_max, self.delta_min = Utilities.max_min_componentwise(
                    self.time_amplitude_train[self.totalModes:, :],
                    self.num_samples)
                delta_mat = self.time_amplitude_train[self.totalModes:, :]
                Utilities.scaling_componentwise(delta_mat, self.delta_max, self.delta_min)
                self.time_amplitude_train[self.totalModes:, :] = delta_mat

            self.parameter_max, self.parameter_min = Utilities.max_min_componentwise_params(self.parameter_train,
                                                                                            self.num_samples,
                                                                                            self.parameter_train.shape[0])
            Utilities.scaling_componentwise_params(self.parameter_train, self.parameter_max, self.parameter_min,
                                                   self.parameter_train.shape[0])

        # Split the 'self.time_amplitudes_train' matrix into -> 'self.data_train_train' and 'self.data_train_val'
        # Split the 'self.parameter_train' matrix into -> 'self.parameter_train_train' and 'self.parameter_train_val'
        self.data_train_train = self.time_amplitude_train[:, :self.num_training_samples]
        self.data_train_val = self.time_amplitude_train[:, self.num_training_samples:]
        self.parameter_train_train = self.parameter_train[:, :self.num_training_samples]
        self.parameter_train_val = self.parameter_train[:, self.num_training_samples:]

        print('DATA PREPARATION DONE ...\n')

        pass

    def input_pipeline(self):
        print('INPUT PIPELINE BUILD START ...\n')
        # We build our input pipeline with the help of dataloader
        # We transpose our data for simplicity purpose
        self.data_train_train = np.transpose(self.data_train_train)
        self.data_train_val = np.transpose(self.data_train_val)
        self.parameter_train_train = np.transpose(self.parameter_train_train)
        self.parameter_train_val = np.transpose(self.parameter_train_val)

        X_train = torch.from_numpy(self.data_train_train).float()
        y_train = torch.from_numpy(self.parameter_train_train).float()
        X_val = torch.from_numpy(self.data_train_val).float()
        y_val = torch.from_numpy(self.parameter_train_val).float()

        # Reshape the training and validation data into the appropriate shape
        if self.typeConv == '2D':
            X_train = torch.reshape(X_train, shape=[-1, 1, int(np.sqrt(self.N)), int(np.sqrt(self.N))])
            X_val = torch.reshape(X_val, shape=[-1, 1, int(np.sqrt(self.N)), int(np.sqrt(self.N))])
        elif self.typeConv == '1D':
            X_train = torch.reshape(X_train, shape=[-1, 1, self.N])
            X_val = torch.reshape(X_val, shape=[-1, 1, self.N])
        else:
            X_train = torch.reshape(X_train, shape=[-1, 1, self.N])
            X_val = torch.reshape(X_val, shape=[-1, 1, self.N])

        dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
        self.training_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=0)
        dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
        self.validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False,
                                                             num_workers=0)

        print('INPUT PIPELINE BUILD DONE ...\n')

        pass

    def loss_function(self, snapshot_data, criterion_h, criterion_N):
        output = snapshot_data.view(snapshot_data.size(0), -1)

        self.loss_h = criterion_h(output, self.dec)
        self.loss_N = criterion_N(self.enc, self.nn)
        self.loss = self.omega_h * self.loss_h + self.omega_N * self.loss_N

        pass

    def training(self, epochs=10000, save_every=100, print_every=10,
                 log_base_name='', pretrained_weights=''):
        self.data_preparation()
        self.input_pipeline()

        log_folder = self.logs_folder + log_base_name + time.strftime("%Y_%m_%d__%H-%M-%S", time.localtime()) + '/'
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)

        log_folder_trained_model = log_folder + '/trained_models/'
        if not os.path.isdir(log_folder_trained_model):
            os.makedirs(log_folder_trained_model)

        self.tensorboard = SummaryWriter(log_dir=log_folder + '/runs/') if TB_MODE else None
        Utilities.save_codeBase(os.getcwd(), log_folder)

        # Instantiate the model
        if self.typeConv == '2D':
            conv_shape = (int(np.sqrt(self.N)), int(np.sqrt(self.N)))
        elif self.typeConv == '1D':
            conv_shape = self.N
        else:
            conv_shape = self.N
        self.model = ConvAutoEncoderDNN(encoded_dimension=self.n,
                                        conv_shape=conv_shape,
                                        num_params=self.num_parameters,
                                        typeConv=self.typeConv)
        if self.pretrained_load:
            # weights = torch.load(pretrained_weights)
            # self.model.load_state_dict(weights)
            self.model.load_net_weights(pretrained_weights)

        # Instantiate the optimizer
        # self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.9)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        # Instantiate the loss
        criterion_h = torch.nn.L1Loss()
        criterion_N = torch.nn.L1Loss()

        # Here we apply the epoch looping throughout the mini batches
        self.best_so_far = 1e12  # For early stopping criteria
        trigger_times_early_stopping = 0
        patience_early_stopping = self.num_early_stop
        training_loss_log = []
        validation_loss_log = []
        for epoch in range(0, epochs):
            print("[INFO] epoch : {}...".format(epoch))

            # TRAINING SECTION.........................
            start_time = time.time()
            trainLoss_h = 0
            trainLoss_N = 0
            trainLoss = 0
            nBatches = 0
            self.model.train()
            # Loop over the mini batches of data
            for batch_idx, (snapshot_data, parameters) in enumerate(self.training_loader):
                # Clear gradients w.r.t parameters
                self.opt.zero_grad()

                # Forward pass to get the outputs and calculate the loss
                self.enc, self.nn, self.dec = self.model(snapshot_data, parameters)

                self.loss_function(snapshot_data, criterion_h, criterion_N)

                # Perform back propagation and update the model parameters
                self.loss.backward()
                self.opt.step()

                # Updating the training losses and the number of batches
                trainLoss_h += self.loss_h.item()
                trainLoss_N += self.loss_N.item()
                trainLoss += self.loss.item()
                nBatches += 1

            training_loss_log.append([trainLoss / nBatches, epoch])
            if (epoch + 1) % save_every == 0:
                self.model.save_network_weights(filePath=log_folder + 'net_weights/', fileName='epoch_' + str(epoch)
                                                                                               + '.pt')
            if self.tensorboard:
                self.tensorboard.add_scalar('Average train_loss_h', trainLoss_h / nBatches, epoch)
                self.tensorboard.add_scalar('Average train_loss_N', trainLoss_N / nBatches, epoch)
                self.tensorboard.add_scalar('Average train loss', trainLoss / nBatches, epoch)

            # Display model progress on the current training set
            if (epoch + 1) % print_every == 0:
                print('Training batch Info...')
                print('Average loss_h at epoch {0} on training set: {1}'.format(epoch, trainLoss_h / nBatches))
                print('Average loss_N at epoch {0} on training set: {1}'.format(epoch, trainLoss_N / nBatches))
                print('Average loss at epoch {0} on training set: {1}'.format(epoch, trainLoss / nBatches))
                print('Took: {0} seconds'.format(time.time() - start_time))

            # VALIDATION SECTION.............................
            start_time = time.time()
            valLoss_h = 0
            valLoss_N = 0
            valLoss = 0
            nBatches = 0
            input_data = None
            output_data = None
            self.model.eval()
            # Loop over mini batches of data
            with torch.no_grad():
                for batch_idx, (snapshot_data, parameters) in enumerate(self.validation_loader):
                    # Forward pass for the validation data
                    self.enc, self.nn, self.dec = self.model(snapshot_data, parameters)
                    if batch_idx == 0:
                        input_data = snapshot_data.view(snapshot_data.size(0), -1)
                        output_data = self.dec
                    else:
                        input_data = torch.cat((input_data, snapshot_data.view(snapshot_data.size(0), -1)), dim=0)
                        output_data = torch.cat((output_data, self.dec), dim=0)

                    # Calculate the loss function corresponding to the outputs
                    self.loss_function(snapshot_data, criterion_h, criterion_N)

                    # Calculate the validation losses
                    valLoss_h += self.loss_h.item()
                    valLoss_N += self.loss_N.item()
                    valLoss += self.loss.item()
                    nBatches += 1

            # scheduler.step()

            validation_loss_log.append([valLoss / nBatches, epoch])
            reco_error = torch.norm(input_data - output_data, p='fro') / torch.norm(input_data, p='fro')

            # Early stopping
            if (valLoss / nBatches) > self.best_so_far:
                trigger_times_early_stopping += 1
                if trigger_times_early_stopping >= patience_early_stopping:
                    self.model.save_network_weights(filePath=log_folder + 'net_weights/', fileName='best_results.pt')
                    f = open(log_folder + 'net_weights/best_results.txt', 'w')
                    f.write(f"epoch:{epoch}; Error: {reco_error:.3e}")
                    f.close()
                    print('\n')
                    print('Early stopping....')
                    print('Validation loss did not improve from {} thus exiting the epoch '
                          'loop.........'.format(self.best_so_far))
                    break
            else:
                self.best_so_far = valLoss / nBatches
                trigger_times_early_stopping = 0

            if self.tensorboard:
                self.tensorboard.add_scalar('Average val_loss_h', valLoss_h / nBatches, epoch)
                self.tensorboard.add_scalar('Average val_loss_N', valLoss_N / nBatches, epoch)
                self.tensorboard.add_scalar('Average val loss', valLoss / nBatches, epoch)

                self.tensorboard.add_scalar('Relative reconstruction error', reco_error, epoch)

            # Display model progress on the current validation set
            if (epoch + 1) % print_every == 0:
                print('Validation batch Info...')
                print('Average loss_h at epoch {0} on validation set: {1}'.format(epoch, valLoss_h / nBatches))
                print('Average loss_N at epoch {0} on validation set: {1}'.format(epoch, valLoss_N / nBatches))
                print('Average loss at epoch {0} on validation set: {1}'.format(epoch, valLoss / nBatches))
                print('Took: {0} seconds\n'.format(time.time() - start_time))

        if self.tensorboard:
            self.tensorboard.close()

        # Save the model
        torch.save(self.model, log_folder_trained_model + 'model.pth')

        # Save the scaling factors for testing the network
        scaling = [self.snapshot_max, self.snapshot_min, self.delta_max, self.delta_min,
                   self.parameter_max, self.parameter_min]
        log_folder_variables = log_folder + '/variables/'
        if not os.path.isdir(log_folder_variables):
            os.makedirs(log_folder_variables)
        np.save(log_folder_variables + 'scaling.npy', scaling, allow_pickle=True)

        return self.model


