from POD_DL_ROM_LIB import Helper
import numpy as np
import time
import torch
import os
from POD_DL_ROM_LIB.NetworkModel import ConvAutoEncoderDNN

try:
    import tensorboard
except ImportError as e:
    TB_MODE = False
else:
    TB_MODE = True
    from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingFramework(object):
    def __init__(self, params, device=DEVICE, log_folder='./training_results_local/') -> None:

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
        self.num_training_samples = int(0.8 * self.num_samples)

    def data_preparation(self):
        print('DATA PREPARATION START...\n')

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
        print('Permuting the columns of SNAPSHOT TRAINING MATRIX and PARAMETER TRAINING MATRIX...\n')
        perm_id = np.random.RandomState(seed=42).permutation(self.snapshot_train.shape[1])
        self.snapshot_train = self.snapshot_train[:, perm_id]
        self.parameter_train = self.parameter_train[:, perm_id]

        print('Splitting the SNAPSHOT TRAINING MATRIX...\n')
        # Split the 'self.snapshot_train' matrix into -> 'self.snapshot_train_train' and 'self.snapshot_train_val'
        self.snapshot_train_train = np.zeros((self.num_dimension * self.N, self.num_training_samples))
        self.snapshot_train_val = np.zeros((self.num_dimension * self.N, self.snapshot_train.shape[1] -
                                            self.num_training_samples))

        print('Compute the intrinsic coordinate matrix for SNAPSHOT TRAINING MATRIX and SNAPSHOT VALIDATION MATRIX...')
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
        print('took:', time.time() - start_time, ' secs...\n')

        # Normalize the data in 'self.snapshot_train_train' and 'self.snapshot_train_val'
        start_time = time.time()
        if self.scaling:
            print(
                'Normalize the training and validation INTRINSIC COORDINATE MATRICES and PARAMETER TRAINING MATRIX...')
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
        print('took: ', time.time() - start_time, ' secs...\n')

        print('Splitting the PARAMETER TRAINING MATRIX...\n')
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

    def training(self, epochs=10000, val_every=1, save_every=1, log_base_name=''):
        self.data_preparation()
        self.input_pipeline()

        log_folder = self.logs_folder + log_base_name + time.strftime("%Y_%m_%d__%H-%M", time.localtime()) + '/'
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)

        self.tensorboard = SummaryWriter() if TB_MODE else None
        # webbrowser.open('http://localhost:6006')
        # os.system('tensorboard --logdir=runs')
        Helper.save_codeBase(os.getcwd(), log_folder)

        # Instantiate the model
        # self.model = DFNCAE(self.n, self.num_parameters, self.N)
        self.model = ConvAutoEncoderDNN()
        # print(self.model)

        # Save the model graph in tensorboard
        # trainWriter.add_graph(self.model, self.inputsForGraphVis)

        # Instantiate the optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # print(len(list(self.model.parameters())))
        # print(list(self.model.parameters())[0].size())

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
                # print(self.enc.size()), print(self.nn.size()), print(self.dec.size())
                # print(type(self.enc)), print(type(self.nn)), print(type(self.dec))
                # print(tmp[0].size())

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
            print('Training batch Info...')
            print('Average loss_h at epoch {0} on training set: {1}'.format(epoch, trainLoss_h / nBatches))
            print('Average loss_N at epoch {0} on training set: {1}'.format(epoch, trainLoss_N / nBatches))
            print('Average loss at epoch {0} on training set: {1}'.format(epoch, trainLoss / nBatches))
            print('Took: {0} seconds'.format(time.time() - start_time))

            if self.tensorboard:
                self.tensorboard.add_scalar('Average train_loss_h', trainLoss_h / nBatches, epoch)
                self.tensorboard.add_scalar('Average train_loss_N', trainLoss_N / nBatches, epoch)
                self.tensorboard.add_scalar('Average train loss', trainLoss / nBatches, epoch)

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

            if (epoch + 1) % val_every:
                validation_loss_log.append([valLoss / nBatches, epoch])
                reco_error = torch.norm(input_data - output_data, p='fro') / torch.norm(input_data, p='fro')
                if reco_error < self.best_so_far:
                    self.best_so_far = reco_error
                    self.model.save_network_weights(filePath=log_folder + 'net_weights/', fileName='best_results.pt')
                    f = open(log_folder + 'net_weights/best_results.txt', 'w')
                    f.write(f"epoch:{epoch}; Error: {reco_error:.3e}")
                    f.close()
                if self.tensorboard:
                    pass  # TO BE DONE#####################################
            # Display model progress on the current validation set
            print('Validation batch Info...')
            print('Average loss_h at epoch {0} on validation set: {1}'.format(epoch, valLoss_h / nBatches))
            print('Average loss_N at epoch {0} on validation set: {1}'.format(epoch, valLoss_N / nBatches))
            print('Average loss at epoch {0} on validation set: {1}'.format(epoch, valLoss / nBatches))
            print('Took: {0} seconds\n'.format(time.time() - start_time))

            if self.tensorboard:
                self.tensorboard.add_scalar('Average val_loss_h', valLoss_h / nBatches, epoch)
                self.tensorboard.add_scalar('Average val_loss_N', valLoss_N / nBatches, epoch)
                self.tensorboard.add_scalar('Average val loss', valLoss / nBatches, epoch)

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

        log_folder_trained_model = './trained_models/'
        if not os.path.isdir(log_folder_trained_model):
            os.makedirs(log_folder_trained_model)

        torch.save(self.model, log_folder_trained_model + 'model.pth')

        pass