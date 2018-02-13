from abc import ABCMeta, abstractmethod
from Conllu_Manager import Conllu_Manager
from KerasManager import KerasManager
import data_helper as dh


class AbstractPOSTaggerTrainer:
    __metaclass__ = ABCMeta

    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir

    @abstractmethod
    def load_resources(self):
        pass

    @abstractmethod
    def train(self, training_path):
        """
        Train the keras model from the training data.

        :param training_path: the path to training file
        :return: the keras model of type Sequential
        """
        pass


class POSTaggerTrainer(AbstractPOSTaggerTrainer):
    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir
        self.model = None

    def load_resources(self):
        pass

    def train(self, training_path):
        modelName1 = 'BLSTM_EMB'
        # modelName2 = 'BLSTM_W2V'
        epochs = 25
        # Train
        train = Conllu_Manager('train', dataSetPath=training_path, embedding='mine')

        x, y = train.generateForOnlineTrainingXY()

        feature = len(train.w2i)
        # syn0, _ = dh.create_word2vec_data(word2index=False)
        if not self.model:
            self.km = KerasManager(modelName1)
            # self.km.BLSTM_MYEMB(feature)
            self.km.Extrapoling_Word2vec_words(syn)
            # self.km.BLSTM_W2V(syn0)
            self.km.printModel()
        else:
            self.km = KerasManager(modelName1, self.model)

        self.model = self.km.trainOnline(x, y, epochs)

        return self.model
