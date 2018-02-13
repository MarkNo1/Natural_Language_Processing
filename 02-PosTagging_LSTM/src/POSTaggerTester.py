from abc import ABCMeta, abstractmethod
from Conllu_Manager import Conllu_Manager
from tqdm import tqdm
from nltk import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import os


class AbstractPOSTaggerTester:
    __metaclass__ = ABCMeta

    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir

    @abstractmethod
    def load_resources(self):
        pass

    @abstractmethod
    def test(self, lstm_pos_tagger, test_file_path):
        """
        Test the lstm_pos_tagger against the gold standard.

        :param lst_pos_tagger: an istance of AbstractLSTMPOSTagger that has to be tested.
        :param test_file_path: a path to the gold standard file.

        :return: a dictionary that has as keys 'precision', 'recall',
        'coverage' and 'f1' and as associated value their respective values.

        Additional info:
        - Precision has to be computed as the number of correctly predicted
          pos tag over the number of predicted pos tags.
        - Recall has to be computed as the number of correctly predicted
          pos tag over the number of items in the gold standard
        - Coverage has to be computed as the number of predicted pos tag over
          the number of items in the gold standard
        - F1 has to be computed as the armonic mean between precision
          and recall (2* P * R / (P + R))
        """
        pass


class POSTaggerTester(AbstractPOSTaggerTester):
    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir

    def load_resources(self):
        pass

    def test(self, lstm_pos_tagger, test_file_path):
        plot_confusion_matrix = False
        test = Conllu_Manager('test', embedding='mine', dataSetPath=test_file_path)
        posTaggedSentenceFilePath = os.path.join(os.getcwd(), 'output', 'pos_tagged_sentences.txt')
        resultFilePath = os.path.join(os.getcwd(), 'output', 'result.txt')
        fp_post_tagged = open(posTaggedSentenceFilePath, 'w')
        fp_result = open(resultFilePath, 'w')
        x, y = test.getSentences()
        y_true = []
        for i in range(len(y)):
            y_true.extend(y[i].ravel())

        y_pred = []
        t = tqdm(range(len(x)))
        for i in t:
            y_sentence_pred = lstm_pos_tagger.predict(x[i])
            y_pred.extend(y_sentence_pred)
            sentence_to_write = ''
            tag_predicted_to_write = ''
            for w in x[i]:
                sentence_to_write += '{} '.format(w)
            for t in y_sentence_pred:
                tag_predicted_to_write += '{} '.format(t)
            fp_post_tagged.write('{}\n'.format(sentence_to_write))
            fp_post_tagged.write('{}\n'.format(tag_predicted_to_write))
            P_, R_, F1_, S_ = precision_recall_fscore_support(y[i], y_sentence_pred, average='micro')
            fp_result.write('Precision: {} , Recall {} , F1 {} , Coverage {} \n'.format(P_, R_, F1_, 1))

        fp_post_tagged.close()
        fp_result.close()

        print(ConfusionMatrix(y_true, y_pred))
        P, R, F1, S = precision_recall_fscore_support(y_true, y_pred, average='micro')

        if plot_confusion_matrix:
            labels_name = []
            for key in test.t2i:
                labels_name.append(key)
            cnf_matrix = confusion_matrix(y_true, y_pred)
            cnf_matrix = cnf_matrix.astype('float64') / cnf_matrix.sum(axis=1)[:, np.newaxis]
            for x in range(0, cnf_matrix.shape[0]):
                for y in range(0, cnf_matrix.shape[1]):
                    if cnf_matrix[x][y] != int(cnf_matrix[x][y]):
                        cnf_matrix[x][y] = round(cnf_matrix[x][y], 3)
            # Plot
            plt.figure(figsize=(15, 15))
            df_cm = pd.DataFrame(cnf_matrix, index=labels_name, columns=labels_name)
            sn.set(font_scale=0.85)
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Blues, square=True, linewidths=.1)
            figurePath = os.path.join(os.getcwd(), 'latec', 'confusionMatrix.png')
            plt.savefig(figurePath, dpi=300)
            plt.close()
        return dict(precision=P, recall=R, coverage=S, f1=F1)
