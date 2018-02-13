from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, InputLayer
from keras.layers import Dropout, Embedding, Masking, Conv1D, MaxPooling1D, Flatten, GRU
from keras.losses import categorical_crossentropy
from keras.initializers import glorot_uniform
from keras.regularizers import l2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from keras.utils import plot_model


class KerasManager:
    def __init__(self, modelName=None, model=None):
        self.modelName = modelName
        self.initParameter()
        self.model = model

    def initParameter(self):
        self.time_series = None
        self.embedding = 100

        self.m_loss = []
        self.m_acc = []

        # Hidden Unints
        self.lstm_hidden_1 = 152
        self.lstm_hidden_2 = 50
        self.dense_hidden_1 = 20
        self.dense_hidden_2 = 100
        self.softmax_output = 17

        # Dropout
        self.lstm_drop_1 = 0.5
        self.lstm_drop_2 = 0.5
        self.dense_drop_1 = 0.5
        self.dense_drop_2 = 0.5

        # Regularization
        self.weights_decay = 1e-8
        self.optimizer = 'rmsprop'

    def setModel(self, model):
        self.model = model

    def getModel(self):
        return self.model

    def printModel(self):
        print(self.model.summary())

    def Embedding_layer(self, features_dim):
        return Embedding(input_dim=features_dim,
                         output_dim=self.embedding,
                         mask_zero=False,
                         input_length=None)

    def Embedding_layer_w2v(self, syn0):
        return Embedding(input_dim=syn0.shape[0],
                         output_dim=syn0.shape[1],
                         mask_zero=False,
                         input_length=None,
                         trainable=False)

    def Lstm_layer(self, hidden):
        return LSTM(hidden,
                    kernel_initializer=glorot_uniform(1),
                    return_sequences=True,
                    kernel_regularizer=l2(self.weights_decay))

    def Lstm_layer_pca(self, hidden, input_dim):
        return LSTM(hidden,
                    kernel_initializer=glorot_uniform(1),
                    return_sequences=True,
                    kernel_regularizer=l2(self.weights_decay),
                    input_shape=(None, None, input_dim))

    def TimeDistributed_layer(self, hidden):
        return TimeDistributed(Dense(hidden,
                                     kernel_initializer=glorot_uniform(1),
                                     activation='relu',
                                     kernel_regularizer=l2(self.weights_decay),
                                     bias_regularizer=l2(self.weights_decay)))

    def BLSTM_W2V(self, syn0):
        model = Sequential()
        model.add(InputLayer(input_shape=(None,)))
        model.add(self.Embedding_layer_w2v(syn0))
        model.add(Bidirectional(self.Lstm_layer(self.lstm_hidden_1)))
        model.add(Dropout(self.lstm_drop_1))
        model.add(self.TimeDistributed_layer(self.dense_hidden_1))
        model.add(Dropout(self.dense_drop_1))
        model.add(Dense(self.softmax_output, activation='softmax', kernel_initializer=glorot_uniform(1)))
        model.compile(loss=categorical_crossentropy, metrics=['accuracy'], optimizer=self.optimizer)
        self.model = model

    def Extrapoling_Word2vec_words(self, syn0):
        model = Sequential()
        model.add(InputLayer(input_shape=(None,)))
        model.add(self.Embedding_layer_w2v(syn0))
        model.compile(loss=categorical_crossentropy, optimizer=self.optimizer)
        self.model = model

    def BLSTM_MYEMB(self, features_dim):
        model = Sequential()
        model.add(InputLayer(input_shape=(None,)))
        model.add(self.Embedding_layer(features_dim))
        model.add(Bidirectional(self.Lstm_layer(self.lstm_hidden_1)))
        model.add(Dropout(self.lstm_drop_1))
        model.add(self.TimeDistributed_layer(self.dense_hidden_1))
        model.add(Dropout(self.dense_drop_1))
        model.add(Dense(self.softmax_output, activation='softmax', kernel_initializer=glorot_uniform(1)))

        model.compile(loss=categorical_crossentropy, metrics=['accuracy'], optimizer=self.optimizer)
        self.model = model

    def BLSTM_MyRealEmbeddingWithPCADim(self, features_dim):
        model = Sequential()
        model.add(InputLayer(input_shape=(None,)))
        model.add(Bidirectional(self.Lstm_layer_pca(self.lstm_hidden_1, 20)))
        model.add(Dropout(self.lstm_drop_1))
        model.add(self.TimeDistributed_layer(self.dense_hidden_1))
        model.add(Dropout(self.dense_drop_1))
        model.add(Dense(self.softmax_output, activation='softmax', kernel_initializer=glorot_uniform(1)))

        model.compile(loss=categorical_crossentropy, metrics=['accuracy'], optimizer=self.optimizer)
        self.model = model

    def trainOnline(self, x, y, epoch):
        iteration = tqdm(range(len(x)), desc='Iteration', leave=True, ascii=True)
        result = []
        for _ in range(epoch):
            for i in iteration:
                loss, acc = self.model.train_on_batch(x[i], y[i])
                result.append([loss, acc])
                mean = np.asarray(result)
                self.m_loss.append(mean.mean(axis=0)[0])
                self.m_acc.append(mean.mean(axis=0)[1])
                iteration.set_description('Iter: {} - loss: {} - acc: {}'.format(i,
                                                                                 self.m_loss[-1], self.m_acc[-1]))
        return self.model

    def plotLoss(self):
        plt.figure()
        plt.plot(self.m_loss)
        plt.show()

    def plotAcc(self):
        plt.figure()
        plt.plot(self.m_acc)
        plt.show()

    def saveModelPicture(self):
        picturePath = os.path.join(os.getcwd(), 'latec', 'Model-{}.png'.format(self.modelName))
        plot_model(self.model, to_file=picturePath, show_shapes=True)
