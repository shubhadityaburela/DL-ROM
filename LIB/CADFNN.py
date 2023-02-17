import torch
import os
import numpy as np

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
