from abc import ABCMeta, abstractmethod
import data_helper as dh
from Conllu_Manager import Conllu_Manager


class AbstractLSTMPOSTagger:
    __metaclass__ = ABCMeta

    def __init__(self, model, resource_dir=None):
        self._model = model
        self._resource_dir = resource_dir

    @abstractmethod
    def load_resources(self):
        pass

    def get_model(self):
        return self._model

    @abstractmethod
    def predict(self, sentence):
        """
        predict the pos tags for each token in the sentence.
        :param sentence: a list of tokens.
        :return: a list of pos tags (one for each input token).
        """
        pass


class LSTMPOSTagger(AbstractLSTMPOSTagger):
    def __init__(self, model, resource_dir=None):
        self._model = model
        self._resource_dir = resource_dir

    def get_model(self):
        return self._model

    def load_resources(self):
        self.converter = Conllu_Manager('converter', embedding='mine')

    def predict(self, sentence):
        sentenceToPredict = self.converter.sentenceToEmb(sentence)
        return dh.sentence_prob_to_tag(self._model.predict(sentenceToPredict)[0], self.converter.i2t)
