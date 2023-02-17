import torch
import os
import numpy as np
import Utilities
import time

try:
    import tensorboard
except ImportError as e:
    TB_MODE = False
else:
    TB_MODE = True
    from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Base(torch.nn.Module):
    def __init__(self, activation=torch.nn.LeakyReLU):
        super(Base, self).__init__()
        self.activation = activation()

    def save_network_weights(self, filePath, fileName):
        if not os.path.isdir(filePath):
            os.makedirs(filePath)
        torch.save(self.state_dict(), filePath + fileName)

    def load_net_weights(self, filePath):
        self.load_state_dict(torch.load(filePath, map_location=DEVICE))


class DeepFeedForwardNetwork(Base):
    def __init__(self, encoded_dimension, num_params, f=torch.nn.LeakyReLU):
        super(self.__class__, self).__init__()

        # Here we define the hyperparameter required for the Deep Feed-forward neural network
        self.num_params = num_params  # Number of parameters for the problem
        self.n = encoded_dimension  # Encoded dimension for the DFNN

        self.ff1 = torch.nn.Linear(in_features=self.num_params, out_features=10)
        torch.nn.init.xavier_uniform_(self.ff1.weight)
        self.act1 = torch.nn.LeakyReLU()
        self.ff1b = torch.nn.BatchNorm1d(10)

        self.ff2 = torch.nn.Linear(in_features=10, out_features=25)
        torch.nn.init.xavier_uniform_(self.ff2.weight)
        self.act2 = torch.nn.LeakyReLU()
        self.ff2b = torch.nn.BatchNorm1d(25)

        self.ff3 = torch.nn.Linear(in_features=25, out_features=10)
        torch.nn.init.xavier_uniform_(self.ff3.weight)
        self.act3 = torch.nn.LeakyReLU()
        self.ff3b = torch.nn.BatchNorm1d(10)

        self.ff4 = torch.nn.Linear(in_features=10, out_features=self.n)
        torch.nn.init.xavier_uniform_(self.ff4.weight)
        self.actlast = f()

    def forward(self, y, apply_f=False, return_nn=False):
        nn = self.ff1b(self.act1(self.ff1(y)))
        nn = self.ff2b(self.act2(self.ff2(nn)))
        nn = self.ff3b(self.act3(self.ff3(nn)))
        nn = self.ff4(nn)

        # No activation and no softmax at the end
        if apply_f:
            if return_nn:
                return nn, self.actlast(nn)
            else:
                return self.actlast(nn)
        else:
            return nn


class DeepFeedForwardNetworkSequential(Base):
    def __init__(self, encoded_dimension, num_params, f=torch.nn.LeakyReLU):
        super(self.__class__, self).__init__()

        # Here we define the hyperparameter required for the Deep Feed-forward neural network
        self.num_params = num_params  # Number of parameters for the problem
        self.n = encoded_dimension  # Encoded dimension for the DFNN

        self.main = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_params, out_features=12),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(12),

            torch.nn.Linear(in_features=12, out_features=36),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(36),

            torch.nn.Linear(in_features=36, out_features=12),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(12),

            torch.nn.Linear(in_features=12, out_features=self.n)
        )

        self.actlast = f()

    def forward(self, y, apply_f=False):

        # No activation and no softmax at the end
        if apply_f:
            return self.actlast(self.main(y))
        else:
            return self.main(y)


class ConvEncoder(Base):
    def __init__(self, encoded_dimension, conv_shape=(16, 16), input_norm=torch.nn.InstanceNorm2d):
        super(self.__class__, self).__init__()

        # Here we define the hyperparameter required for the Convolutional Encoder architecture
        self.k = 5  # Size of the convolutional kernel
        self.n = encoded_dimension  # Encoded dimension for the convolutional encoder
        self.conv_shape_list = []  # The list storing the convolutional shapes of the inputs and outputs

        # Encoder layers which take in the input of size corresponding to 'N' and reduce it to encoded dimension 'n'
        self.conv_shape = conv_shape
        self.input_norm = input_norm(1)

        self.conv_shape_list.append(self.conv_shape)

        self.padding = (2, 2)
        self.kernel_size = (self.k, self.k)
        self.stride = (1, 1)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding)
        self.conv1b = torch.nn.BatchNorm2d(8)
        self.conv_shape = [np.floor(
            (self.conv_shape[i] + 2 * self.padding[i] - self.kernel_size[i]) / self.stride[i] + 1)
            for i in range(len(conv_shape))]

        self.conv_shape_list.append(self.conv_shape)

        self.padding = (2, 2)
        self.kernel_size = (self.k, self.k)
        self.stride = (2, 2)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding)
        self.conv2b = torch.nn.BatchNorm2d(16)
        self.conv_shape = [np.floor(
            (self.conv_shape[i] + 2 * self.padding[i] - self.kernel_size[i]) / self.stride[i] + 1)
            for i in range(len(conv_shape))]

        self.conv_shape_list.append(self.conv_shape)

        self.padding = (2, 2)
        self.kernel_size = (self.k, self.k)
        self.stride = (2, 2)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding)
        self.conv3b = torch.nn.BatchNorm2d(32)
        self.conv_shape = [np.floor(
            (self.conv_shape[i] + 2 * self.padding[i] - self.kernel_size[i]) / self.stride[i] + 1)
            for i in range(len(conv_shape))]

        self.conv_shape_list.append(self.conv_shape)

        self.padding = (2, 2)
        self.kernel_size = (self.k, self.k)
        self.stride = (2, 2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding)
        self.conv4b = torch.nn.BatchNorm2d(64)
        self.conv_shape = [np.floor(
            (self.conv_shape[i] + 2 * self.padding[i] - self.kernel_size[i]) / self.stride[i] + 1)
            for i in range(len(conv_shape))]

        self.conv_shape_list.append(self.conv_shape)

        num_channels_last_layer = 64
        feature_dim_encoding = self.conv_shape[0] * self.conv_shape[1] * num_channels_last_layer
        self.fc1 = torch.nn.Linear(in_features=int(feature_dim_encoding), out_features=64)
        self.fc1b = torch.nn.BatchNorm1d(64)

        self.fc2 = torch.nn.Linear(in_features=64, out_features=self.n)

    def forward(self, x):
        x = self.input_norm(x)

        # Feature generation
        x = self.conv1b(self.activation(self.conv1(x)))
        x = self.conv2b(self.activation(self.conv2(x)))
        x = self.conv3b(self.activation(self.conv3(x)))
        x = self.conv4b(self.activation(self.conv4(x)))

        enc = x.view(x.size(0), -1)
        enc = self.fc1b(self.activation(self.fc1(enc)))
        enc = self.fc2(enc)

        return enc


class ConvDecoder(Base):
    def __init__(self, encoded_dimension, f=torch.nn.Sigmoid, conv_shape=None, lam=1, init_zeros=False):
        super(self.__class__, self).__init__()

        # Here we define the hyperparameter required for the Convolutional Encoder architecture
        self.k = 5  # Size of the convolutional kernel
        self.n = encoded_dimension  # Encoded dimension for the convolutional encoder
        self.lam = lam
        self.conv_shape_list = conv_shape

        # Decoder layers that take in the dimension 'n' and expands it to N
        self.fc1_t = torch.nn.Linear(in_features=self.n, out_features=64)
        self.fc1b_t = torch.nn.BatchNorm1d(64)

        self.fc2_t = torch.nn.Linear(in_features=64, out_features=64 * int(self.conv_shape_list[-1][0]) * int(self.conv_shape_list[-1][1]))

        self.conv1b_t = torch.nn.BatchNorm2d(64)
        self.stride = (1, 1)
        self.kernel_size = (self.k, self.k)
        p1 = np.floor(((self.conv_shape_list[-1][0] - 1) * self.stride[0] + self.k - self.conv_shape_list[-2][0]) / 2)
        p2 = np.floor(((self.conv_shape_list[-1][1] - 1) * self.stride[1] + self.k - self.conv_shape_list[-2][1]) / 2)
        self.padding = (int(p1), int(p2))
        self.conv1_t = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=self.kernel_size,
                                                stride=self.stride, padding=self.padding)

        self.conv2b_t = torch.nn.BatchNorm2d(64)
        self.stride = (1, 1)
        self.kernel_size = (self.k, self.k)
        p1 = np.floor(((self.conv_shape_list[-2][0] - 1) * self.stride[0] + self.k - self.conv_shape_list[-3][0]) / 2)
        p2 = np.floor(((self.conv_shape_list[-2][1] - 1) * self.stride[1] + self.k - self.conv_shape_list[-3][1]) / 2)
        self.padding = (int(p1), int(p2))
        self.conv2_t = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(self.k, self.k),
                                                stride=self.stride, padding=self.padding)

        self.conv3b_t = torch.nn.BatchNorm2d(32)
        self.stride = (2, 2)
        self.kernel_size = (self.k, self.k)
        p1 = np.floor(((self.conv_shape_list[-3][0] - 1) * self.stride[0] + self.k - self.conv_shape_list[-4][0]) / 2)
        p2 = np.floor(((self.conv_shape_list[-3][1] - 1) * self.stride[1] + self.k - self.conv_shape_list[-4][1]) / 2)
        self.padding = (int(p1), int(p2))
        self.conv3_t = torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(self.k, self.k),
                                                stride=self.stride, padding=self.padding)

        self.conv4b_t = torch.nn.BatchNorm2d(16)
        self.stride = (2, 2)
        self.kernel_size = (self.k, self.k)
        p1 = np.floor(((self.conv_shape_list[-4][0] - 1) * self.stride[0] + self.k - self.conv_shape_list[-5][0]) / 2)
        p2 = np.floor(((self.conv_shape_list[-4][1] - 1) * self.stride[1] + self.k - self.conv_shape_list[-5][1]) / 2)
        self.padding = (int(p1), int(p2))
        self.conv4_t = torch.nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(self.k, self.k),
                                                stride=self.stride, padding=self.padding)

        # Adaptive pooling to get the desired output size
        self.adaPool = torch.nn.AdaptiveAvgPool2d((self.conv_shape_list[-5][0], self.conv_shape_list[-5][1]))

        self.f = f()

    def forward(self, x, apply_f=True, return_dec=False):
        x = self.fc1b_t(self.activation(self.fc1_t(x)))
        x = self.activation(self.fc2_t(x))

        x_s = torch.reshape(x, shape=[x.size(0), 64, -1])
        dec = self.conv1b_t(torch.reshape(x, shape=[x.size(0), 64, int(np.sqrt(x_s.shape[2])),
                                                    int(np.sqrt(x_s.shape[2]))]))

        # Feature generation
        dec = self.conv2b_t(self.activation(self.conv1_t(dec)))
        dec = self.conv3b_t(self.activation(self.conv2_t(dec)))
        dec = self.conv4b_t(self.activation(self.conv3_t(dec)))
        dec = self.conv4_t(dec)
        dec = self.adaPool(dec)

        if apply_f:
            if return_dec:
                return dec, self.f(dec/self.lam)
            else:
                return self.f(dec/self.lam)
        else:
            return dec


class ConvEncoder1D(Base):
    def __init__(self, encoded_dimension, conv_shape=None, input_norm=torch.nn.InstanceNorm1d):
        super(self.__class__, self).__init__()

        # Here we define the hyperparameter required for the Convolutional Encoder architecture
        self.k = 3  # Size of the convolutional kernel
        self.n = encoded_dimension  # Encoded dimension for the convolutional encoder
        self.conv_shape_list = []  # The list storing the convolutional shapes of the inputs and outputs

        # Encoder layers which take in the input of size corresponding to 'N' and reduce it to encoded dimension 'n'
        self.conv_shape = conv_shape
        self.input_norm = input_norm(1)

        self.conv_shape_list.append(self.conv_shape)

        self.padding = 0
        self.kernel_size = self.k
        self.stride = 1
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding)
        self.conv1b = torch.nn.BatchNorm1d(8)
        self.conv_shape = np.floor((self.conv_shape + 2 * self.padding - self.kernel_size) / self.stride + 1)

        self.conv_shape_list.append(self.conv_shape)

        self.padding = 1
        self.kernel_size = self.k
        self.stride = 1
        self.conv2 = torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding)
        self.conv2b = torch.nn.BatchNorm1d(16)
        self.conv_shape = np.floor((self.conv_shape + 2 * self.padding - self.kernel_size) / self.stride + 1)

        self.conv_shape_list.append(self.conv_shape)

        num_channels_last_layer = 16
        feature_dim_encoding = self.conv_shape * num_channels_last_layer
        self.fc1 = torch.nn.Linear(in_features=int(feature_dim_encoding), out_features=16)
        self.fc1b = torch.nn.BatchNorm1d(16)

        self.fc2 = torch.nn.Linear(in_features=16, out_features=self.n)

    def forward(self, x):
        x = self.input_norm(x)

        # Feature generation
        x = self.conv1b(self.activation(self.conv1(x)))
        x = self.conv2b(self.activation(self.conv2(x)))

        enc = x.view(x.size(0), -1)
        enc = self.fc1b(self.activation(self.fc1(enc)))

        enc = self.fc2(enc)

        return enc


class ConvDecoder1D(Base):
    def __init__(self, encoded_dimension, f=torch.nn.LeakyReLU, conv_shape=None, lam=1, init_zeros=False):
        super(self.__class__, self).__init__()

        # Here we define the hyperparameter required for the Convolutional Encoder architecture
        self.k = 3  # Size of the convolutional kernel
        self.n = encoded_dimension  # Encoded dimension for the convolutional encoder
        self.lam = lam
        self.conv_shape_list = conv_shape

        # Decoder layers that take in the dimension 'n' and expands it to N
        self.fc1_t = torch.nn.Linear(in_features=self.n, out_features=16)
        self.fc1b_t = torch.nn.BatchNorm1d(16)

        self.fc2_t = torch.nn.Linear(in_features=16, out_features=16 * int(self.conv_shape_list[-1]))

        self.conv1b_t = torch.nn.BatchNorm1d(16)
        self.kernel_size = self.k
        self.stride = 1
        p = np.floor(((self.conv_shape_list[-1] - 1) * self.stride + self.k - self.conv_shape_list[-2]) / 2)
        while p < 0:
            self.stride = self.stride + 1
            p = np.floor(((self.conv_shape_list[-1] - 1) * self.stride + self.k - self.conv_shape_list[-2]) / 2)
        self.padding = int(p)
        self.conv1_t = torch.nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=self.kernel_size,
                                                stride=self.stride, padding=self.padding)

        self.conv2b_t = torch.nn.BatchNorm1d(8)
        self.kernel_size = self.k
        self.stride = 1
        p = np.floor(((self.conv_shape_list[-2] - 1) * self.stride + self.k - self.conv_shape_list[-3]) / 2)
        while p < 0:
            self.stride = self.stride + 1
            p = np.floor(((self.conv_shape_list[-2] - 1) * self.stride + self.k - self.conv_shape_list[-3]) / 2)
        self.padding = int(p)
        self.conv2_t = torch.nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=self.kernel_size,
                                                stride=self.stride, padding=self.padding)

        # Adaptive pooling to get the desired output size
        self.adaPool = torch.nn.AdaptiveAvgPool1d(self.conv_shape_list[-3])

        self.f = f()

    def forward(self, x, apply_f=True, return_dec=False):
        x = self.fc1b_t(self.activation(self.fc1_t(x)))
        x = self.activation(self.fc2_t(x))

        dec = torch.reshape(x, shape=[x.size(0), 16, -1])
        dec = self.conv1b_t(dec)

        # Feature generation
        dec = self.conv2b_t(self.activation(self.conv1_t(dec)))
        dec = self.conv2_t(dec)
        dec = self.adaPool(dec)

        if apply_f:
            if return_dec:
                return dec, self.f(dec/self.lam)
            else:
                return self.f(dec/self.lam)
        else:
            return dec


class ConvAutoEncoderDNN(Base):
    def __init__(self, encoder=None, df_nn=None, decoder=None, encoded_dimension=4, f=torch.nn.LeakyReLU,
                 conv_shape=16, num_params=2, typeConv='1D'):
        super(self.__class__, self).__init__()

        self.typeConv = typeConv

        if self.typeConv == '2D':
            self.Conv_encoder = ConvEncoder(encoded_dimension, conv_shape) if encoder is None else encoder
        elif self.typeConv == '1D':
            self.Conv_encoder = ConvEncoder1D(encoded_dimension, conv_shape) if encoder is None else encoder
        else:
            self.Conv_encoder = ConvEncoder1D(encoded_dimension, conv_shape) if encoder is None else encoder

        self.Deep_FNN = DeepFeedForwardNetworkSequential(encoded_dimension, num_params) if df_nn is None else df_nn

        if self.typeConv == '2D':
            self.Conv_decoder = ConvDecoder(encoded_dimension, f, conv_shape=self.Conv_encoder.conv_shape_list) \
                if decoder is None else decoder
        elif self.typeConv == '1D':
            self.Conv_decoder = ConvDecoder1D(encoded_dimension, f, conv_shape=self.Conv_encoder.conv_shape_list) \
                if decoder is None else decoder
        else:
            self.Conv_decoder = ConvDecoder1D(encoded_dimension, f, conv_shape=self.Conv_encoder.conv_shape_list) \
                if decoder is None else decoder

    def forward(self, x, y, return_code=False, apply_f=True, return_res=False):
        enc = self.Conv_encoder(x)
        nn = self.Deep_FNN(y)
        dec = self.Conv_decoder(nn, apply_f)

        return enc, nn, dec.view(dec.size(0), -1)

    def forward_test(self, y, return_code=False, apply_f=True, return_res=False):
        nn = self.Deep_FNN(y)
        dec = self.Conv_decoder(nn, apply_f)

        return nn, dec.view(dec.size(0), -1)


class TrainingFramework(object):
    def __init__(self, params, split=0.67, device=DEVICE, log_folder='./training_results_local/') -> None:

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

        # Normalize the wildfire_data
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
        # We transpose our wildfire_data for simplicity purpose
        self.data_train_train = np.transpose(self.data_train_train)
        self.data_train_val = np.transpose(self.data_train_val)
        self.parameter_train_train = np.transpose(self.parameter_train_train)
        self.parameter_train_val = np.transpose(self.parameter_train_val)

        X_train = torch.from_numpy(self.data_train_train).float()
        y_train = torch.from_numpy(self.parameter_train_train).float()
        X_val = torch.from_numpy(self.data_train_val).float()
        y_val = torch.from_numpy(self.parameter_train_val).float()

        # Reshape the training and validation wildfire_data into the appropriate shape
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
            # Loop over the mini batches of wildfire_data
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
            # Loop over mini batches of wildfire_data
            with torch.no_grad():
                for batch_idx, (snapshot_data, parameters) in enumerate(self.validation_loader):
                    # Forward pass for the validation wildfire_data
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