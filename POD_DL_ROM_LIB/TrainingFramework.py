import Helper
import numpy as np
import time
import torch
import os
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
        self.logs_folder = log_folder
        self.tensorboard = None

        self.snapshot_train_val = None
        self.snapshot_train_train = None
        self.parameter_train_val = None
        self.parameter_train_train = None
        self.snapshot_test_test = None

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

        self.inputsForGraphVis = None

        # We perform an 80-20 split for the training and validation set
        self.num_training_samples = int(split * self.num_samples)

    def data_preparation(self):
        print('DATA PREPARATION START...\n')

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

            # We now perform random permutation of the columns of the 'self.snapshot_train' and 'self.parameter_train'
            # to better generalize the split
            perm_id = np.random.RandomState(seed=42).permutation(self.snapshot_train.shape[1])
            self.snapshot_train = self.snapshot_train[:, perm_id]
            self.parameter_train = self.parameter_train[:, perm_id]

            # Split the 'self.snapshot_train' matrix into -> 'self.snapshot_train_train' and 'self.snapshot_train_val'
            self.snapshot_train_train = np.zeros((self.num_dimension * self.N, self.num_training_samples))
            self.snapshot_train_val = np.zeros((self.num_dimension * self.N, self.snapshot_train.shape[1] -
                                                self.num_training_samples))

            # Compute the intrinsic coordinates for 'self.snapshot_train_train' and 'self.snapshot_train_val'
            # by performing a projection onto the reduced basis.
            # self.snapshot_train_train = (self.U)^T x self.snapshot_train
            start_time = time.time()
            self.U_transpose = np.transpose(self.U)  # making it (U^T)
            for i in range(self.num_dimension):
                self.snapshot_train_train[i * self.N:(i + 1) * self.N, :] = np.matmul(
                    self.U_transpose[:, i * self.N_h:(i + 1) * self.N_h],
                    self.snapshot_train[i * self.N_h:(i + 1) * self.N_h, :self.num_training_samples])
                self.snapshot_train_val[i * self.N:(i + 1) * self.N, :] = np.matmul(
                    self.U_transpose[:, i * self.N_h:(i + 1) * self.N_h],
                    self.snapshot_train[i * self.N_h:(i + 1) * self.N_h, self.num_training_samples:])
        else:
            self.snapshot_train_train = np.zeros((self.num_dimension * self.N, self.num_training_samples))
            self.snapshot_train_val = np.zeros(
                (self.num_dimension * self.N, self.num_samples - self.num_training_samples))

            perm_id = np.random.RandomState(seed=42).permutation(self.time_amplitude_train.shape[1])
            self.time_amplitude_train = self.time_amplitude_train[:, perm_id]
            self.parameter_train = self.parameter_train[:, perm_id]

            self.snapshot_train_train = self.time_amplitude_train[:, :self.num_training_samples]
            self.snapshot_train_val = self.time_amplitude_train[:, self.num_training_samples:]

        # Normalize the data in 'self.snapshot_train_train' and 'self.snapshot_train_val'
        if self.scaling:
            snapshot_max, snapshot_min = Helper.max_min_componentwise(self.snapshot_train_train,
                                                                      self.num_training_samples,
                                                                      self.num_dimension, self.N)
            Helper.scaling_componentwise(self.snapshot_train_train, snapshot_max, snapshot_min,
                                         self.num_dimension, self.N)
            Helper.scaling_componentwise(self.snapshot_train_val, snapshot_max, snapshot_min,
                                         self.num_dimension, self.N)

            parameter_max, parameter_min = Helper.max_min_componentwise_params(self.parameter_train,
                                                                               self.num_training_samples,
                                                                               self.parameter_train.shape[0])
            Helper.scaling_componentwise_params(self.parameter_train, parameter_max, parameter_min,
                                                self.parameter_train.shape[0])

        # Split the 'self.parameter_train' matrix into -> 'self.parameter_train_train' and 'self.parameter_train_val'
        self.parameter_train_train = self.parameter_train[:, :self.num_training_samples]
        self.parameter_train_val = self.parameter_train[:, self.num_training_samples:]

        print('DATA PREPARATION DONE ...\n')

        pass

    def input_pipeline(self):
        print('INPUT PIPELINE BUILD START ...\n')
        # We build our input pipeline with the help of dataloader
        # We transpose our data for simplicity purpose
        self.snapshot_train_train = np.transpose(self.snapshot_train_train)
        self.snapshot_train_val = np.transpose(self.snapshot_train_val)
        self.parameter_train_train = np.transpose(self.parameter_train_train)
        self.parameter_train_val = np.transpose(self.parameter_train_val)

        X_train = torch.from_numpy(self.snapshot_train_train).float()
        y_train = torch.from_numpy(self.parameter_train_train).float()
        X_val = torch.from_numpy(self.snapshot_train_val).float()
        y_val = torch.from_numpy(self.parameter_train_val).float()

        # Reshape the training and validation data into the appropriate shape
        if self.typeConv == '2D':
            X_train = torch.reshape(X_train, shape=[-1, self.num_dimension,
                                                    int(np.sqrt(self.N)), int(np.sqrt(self.N))])
            X_val = torch.reshape(X_val, shape=[-1, self.num_dimension,
                                                int(np.sqrt(self.N)), int(np.sqrt(self.N))])
        elif self.typeConv == '1D':
            X_train = torch.reshape(X_train, shape=[-1, self.num_dimension, self.N])
            X_val = torch.reshape(X_val, shape=[-1, self.num_dimension, self.N])
        else:
            X_train = torch.reshape(X_train, shape=[-1, self.num_dimension,
                                                    int(np.sqrt(self.N)), int(np.sqrt(self.N))])
            X_val = torch.reshape(X_val, shape=[-1, self.num_dimension,
                                                int(np.sqrt(self.N)), int(np.sqrt(self.N))])

        dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
        self.training_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=0)
        dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
        self.validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False,
                                                             num_workers=0)

        print('INPUT PIPELINE BUILD DONE ...\n')

        for batch_idx, (X, y) in enumerate(self.training_loader):
            if batch_idx == 0:
                self.inputsForGraphVis = (X, y)
            break

        pass

    def loss_function(self, snapshot_data):
        output = snapshot_data.view(snapshot_data.size(0), -1)
        self.loss_h = (self.omega_h / 2) * torch.mean(torch.sum(torch.pow(output - self.dec, 2), dim=1))
        self.loss_N = (self.omega_N / 2) * torch.mean(torch.sum(torch.pow(self.enc - self.nn, 2), dim=1))
        self.loss = self.loss_h + self.loss_N

        pass

    def training(self, epochs=10000, val_every=100, save_every=100, print_every=10, fig_save_every=500,
                 log_base_name=''):
        self.data_preparation()
        self.input_pipeline()

        log_folder = self.logs_folder + log_base_name + time.strftime("%Y_%m_%d__%H-%M-%S", time.localtime()) + '/'
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)

        log_folder_trained_model = log_folder + '/trained_models/'
        if not os.path.isdir(log_folder_trained_model):
            os.makedirs(log_folder_trained_model)

        self.tensorboard = SummaryWriter(log_dir=log_folder + '/runs/') if TB_MODE else None
        Helper.save_codeBase(os.getcwd(), log_folder)

        # Instantiate the model
        if self.typeConv == '2D':
            conv_shape = (int(np.sqrt(self.N)), int(np.sqrt(self.N)))
        elif self.typeConv == '1D':
            conv_shape = self.N
        else:
            conv_shape = (int(np.sqrt(self.N)), int(np.sqrt(self.N)))
        self.model = ConvAutoEncoderDNN(conv_shape=conv_shape, num_params=self.num_parameters, typeConv=self.typeConv)

        # Instantiate the optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Here we apply the epoch looping throughout the mini batches
        self.best_so_far = 1e12  # For early stopping criteria
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
            for i, (snapshot_data, parameters) in enumerate(self.training_loader):
                # Clear gradients w.r.t parameters
                self.opt.zero_grad()

                # Forward pass to get the outputs and calculate the loss
                self.enc, self.nn, self.dec = self.model(snapshot_data, parameters)

                self.loss_function(snapshot_data)

                # Perform back propagation and update the model parameters
                self.loss.backward()
                self.opt.step()

                # Updating the training losses and the number of batches
                trainLoss_h += self.loss_h
                trainLoss_N += self.loss_N
                trainLoss += self.loss
                nBatches += 1

            training_loss_log.append([trainLoss / nBatches, epoch])
            if (epoch + 1) % save_every == 0:
                self.model.save_network_weights(filePath=log_folder + 'net_weights/', fileName='epoch_' + str(epoch)
                                                                                               + '.pt')
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
            input_data = torch.zeros_like(torch.from_numpy(self.snapshot_train_val).float())
            output_data = torch.zeros_like(input_data)
            self.model.eval()
            # Loop over mini batches of data
            with torch.no_grad():
                for i, (snapshot_data, parameters) in enumerate(self.validation_loader):
                    # Forward pass for the validation data
                    self.enc, self.nn, self.dec = self.model(snapshot_data, parameters)
                    input_data[nBatches * self.batch_size:(nBatches + 1) * self.batch_size, :] = \
                        snapshot_data.view(snapshot_data.size(0), -1)
                    output_data[nBatches * self.batch_size:(nBatches + 1) * self.batch_size, :] = self.dec

                    # Calculate the loss function corresponding to the outputs
                    self.loss_function(snapshot_data)

                    # Calculate the validation losses
                    valLoss_h += self.loss_h
                    valLoss_N += self.loss_N
                    valLoss += self.loss
                    nBatches += 1
                valLoss_mean = valLoss / nBatches

            if (epoch + 1) % val_every == 0:
                validation_loss_log.append([valLoss / nBatches, epoch])
                reco_error = torch.norm(input_data - output_data, p='fro') / torch.norm(input_data, p='fro')
                if reco_error < self.best_so_far:
                    self.best_so_far = reco_error
                    self.model.save_network_weights(filePath=log_folder + 'net_weights/', fileName='best_results.pt')
                    torch.save(self.model, log_folder_trained_model + 'model.pth')
                    f = open(log_folder + 'net_weights/best_results.txt', 'w')
                    f.write(f"epoch:{epoch}; Error: {reco_error:.3e}")
                    f.close()
                if self.tensorboard:
                    if (epoch + 1) % fig_save_every == 0:
                        fig_reco = self.plot_val_idx_reco(torch.transpose(output_data, 0, 1))
                        self.tensorboard.add_figure('reconstruction', fig_reco, global_step=epoch, close=True)
                    self.tensorboard.add_scalar('Average train_loss_h', trainLoss_h / nBatches, epoch)
                    self.tensorboard.add_scalar('Average train_loss_N', trainLoss_N / nBatches, epoch)
                    self.tensorboard.add_scalar('Average train loss', trainLoss / nBatches, epoch)

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

            if valLoss_mean < self.best_so_far:
                self.best_so_far = valLoss_mean
                count = 0
            else:
                count += 1
            # Early stopping criterion
            if count == self.num_early_stop:
                print('Stopped training due to early-stopping cross-validation')
                break
        print('Best loss on validation set: {0}'.format(self.best_so_far))

        if self.tensorboard:
            self.tensorboard.close()

        pass

    def plot_val_idx_reco(self, reco, plot_idx=None):
        plot_idx = np.random.randint(self.snapshot_train_val.transpose().shape[0]) if plot_idx is None else plot_idx
        truth = Helper.to_torch(self.snapshot_train_val.transpose()[[plot_idx], ...], self.device)

        return plot_reconstruction(truth, reco)


def plot_reconstruction(truth, reco, phi=None):
    p1 = (phi is not None) * 1
    fig, axes = plt.subplots(1, 3 + p1, num=0, figsize=[19.2, 3.6])
    dims_remove = [0] * (truth.ndim - 2) + [...]
    if p1:
        pc = axes[0].imshow(phi[dims_remove].to('cpu').T, origin='lower')
        plt.colorbar(pc, ax=axes[0])
        axes[0].set_title('$ \phi $')
        axes[0].axis('equal')
    pc = axes[0 + p1].imshow(reco[dims_remove].to('cpu').T, origin='lower')
    axes[0 + p1].set_title('$ \^q $')
    axes[0 + p1].axis('equal')
    plt.colorbar(pc, ax=axes[0 + p1])
    pc = axes[1 + p1].imshow(truth[dims_remove].to('cpu').T, origin='lower')
    plt.colorbar(pc, ax=axes[1 + p1])
    axes[1 + p1].set_title('$ q $')
    axes[1 + p1].axis('equal')
    pc = axes[2 + p1].imshow((reco - truth)[dims_remove].abs().to('cpu').T, origin='lower')
    plt.colorbar(pc, ax=axes[2 + p1])
    axes[2 + p1].set_title('$| \^q - q |$')
    axes[2 + p1].axis('equal')

    return fig
