from data_helper import create_word2vec_data, c
import pickle as pk
from sklearn.utils import shuffle
import numpy as np
import sys
import os


class Conllu_Manager():
    def __init__(self, managerName=None, embedding=None, dataSetPath=None):
        self.managerName = managerName
        self.dataSetPath = dataSetPath
        self.embedding = embedding
        self.Info()
        if dataSetPath:
            self.dataSentenceDictionary = self.loadConlluDataName(-1)
        self.w2i, self.i2w = self.initWordIndexing()
        self.t2i, self.i2t = self.initLabelsIndexing()

    def reInit(self):
        if self.dataSetPath:
            self.dataSentenceDictionary = self.loadConlluDataName(-1)
        self.w2i, self.i2w = self.initWordIndexing()
        self.t2i, self.i2t = self.initLabelsIndexing()

    def Info(self):
        self.LOG('\n DataSet= {} \n Embeddings= {} \n'.format(
            c.orange(self.dataSetPath), c.orange(self.embedding)))

    def LOG(self, text, logname=True):
        logName = ''
        if logname:
            logName = c.blue('[ConlluManager - {} ] '.format(self.managerName))
        sys.stdout.write(logName + text)

    def LOGResponse(self, val, r=True):
        re = ''
        if r:
            re = '\n'
        if val:
            sys.stdout.write(c.green('\tDONE' + re))
        else:
            sys.stdout.write(c.red('\tERROR' + re))

    def getUniversalPosTag(self):
        return dict(ADJ='adjective', ADP='adposition', ADV='adverb',
                    AUX='auxiliary', CONJ='coordinating conjunction',
                    DET='determiner', INTJ='interjection', NOUN='noun',
                    NUM='numeral', PART='particle', PRON='pronoun',
                    PROPN='proper noun', PUNCT='punctuation', SCONJ='subordinating conjunction',
                    SYM='symbol', VERB='verb', X='other')

    def loadConlluDataName(self, cutting):
        I = c.orange('*loadConlluDataName ')
        self.LOG(I + 'Loading Dictionary ... ')
        """ Extract the sentences and tag_set from data_path
        :param conllu_data_path:
        :return: dict( sentence_number= dic( Sentence:sentence,
                                             word_id: dict(word_id,form,lemma,U-postag,X-postag))))
        """
        INIT_SENTENCE = '#'
        END_SENTENCE = ''
        conllu_dict = dict()

        f = open(self.dataSetPath, 'r')
        data_splitted_by_line = f.read().split('\n')
        sentence = []
        word_id = []
        counter_s = 0
        counter_w = 0

        for idx, line in enumerate(data_splitted_by_line):
            line_separated_by_tab = line.split('\t')

            if line_separated_by_tab[0].startswith(INIT_SENTENCE):
                sentence = line.split('# sentence-text:')[1]

            if line_separated_by_tab[0].isdigit():
                word_id.append(dict(form=line_separated_by_tab[1],
                                    lemma=line_separated_by_tab[2],
                                    u_postag=line_separated_by_tab[3],
                                    x_postag=line_separated_by_tab[4]))
                counter_w += 1

            if line_separated_by_tab[0] is END_SENTENCE:
                conllu_dict['Sentence_' + str(counter_s)] = {'sentence': sentence, 'labels': word_id}
                sentence = []
                word_id = []
                counter_s += 1

            if counter_s == cutting:
                break

        f.close()
        s_to_del = 'Sentence_{}'.format(str(counter_s))
        if s_to_del in conllu_dict:
            del conllu_dict[s_to_del]

        self.LOGResponse(True, r=False)
        self.LOG('  Sentence= {} - Token= {} \n'.format(counter_s, counter_w), logname=False)
        return conllu_dict

    def initWordIndexing(self):
        I = c.orange('*initWordAndLabelIndexing ')
        wordIndexingPath = os.path.join(os.getcwd(), 'resources', 'wordIndexing-{}.data'.format(self.embedding))
        if os.path.exists(wordIndexingPath):
            with open(wordIndexingPath, 'rb') as f:
                data = pk.loads(f.read())
                w2i = data['w2i']
                i2w = data['i2w']
            self.LOG(I + 'Word to index reloaded .')
            self.LOGResponse(True)
        else:
            if self.embedding is 'word2vec':
                _, w2i = create_word2vec_data(weights=False)
                i2w = dict((idx, w) for idx, w in enumerate(w2i))
                self.LOG(I + 'Word2Vec Indexing -> ')
                self.LOGResponse(True)
            elif self.embedding is 'mine':
                self.LOG(I + 'Creating words vocabulary  ... ')
                words = []
                for phrase in self.dataSentenceDictionary:
                    for w_id in self.dataSentenceDictionary[phrase]['labels']:
                        words.append(w_id['form'])
                words_single_appear = sorted(list(set(words)))
                w2i = dict((w, idx + 2) for idx, w in enumerate(words_single_appear))
                w2i['<UNK>'] = 0
                w2i['</s>'] = 1
                i2w = dict((idx + 2, w) for idx, w in enumerate(words_single_appear))
                i2w[0] = '<UNK>'
                i2w[1] = '</s>'
                vocab_size = len(w2i)
                self.LOG(' Vocabulary size ' + str(vocab_size), logname=False)
                self.LOGResponse(True)

            with open(wordIndexingPath, 'wb') as f:
                w2i = pk.dump(dict(w2i=w2i, i2w=i2w), f)
            self.LOG(I + 'Word to index saved .')
            self.LOGResponse(True)

        return w2i, i2w

    def initLabelsIndexing(self):
        I = c.orange('*initLabelsIndexing ')
        tags = self.getUniversalPosTag()
        t2i = dict((tag, idx) for idx, tag in enumerate(tags))
        i2t = dict((idx, tag) for idx, tag in enumerate(tags))
        self.LOG(I + 'Labels Indexing -> ')
        self.LOGResponse(True)
        return t2i, i2t

    def getSentences(self):
        I = c.orange('*getSentences')
        sentence_w = []
        sentence_l = []
        for phrase in self.dataSentenceDictionary:
            w_list = []     # Word
            l_list = []     # Label
            for w_id in self.dataSentenceDictionary[phrase]['labels']:
                w_list.append(w_id['form'])
                l_list.append(w_id['u_postag'])
            sentence_w.append(np.asarray(w_list))
            sentence_l.append(np.asarray(l_list))
        self.LOG(I + ' Words : {} - Labels : {}'.format(str(len(sentence_w)), str(len(sentence_l))))
        self.LOGResponse(True)
        return sentence_w, sentence_l

    # Convert Word to Embedding and Label to tagType : [None:TAG, False:t2i, True: tag vectorized]
    def convertedWordAndLabel(self, tagType=None):
        I = c.orange('*convertedWordAndLabel')
        sentence_w, sentence_l = self.getSentences()
        new_sentence_w = []
        new_sentence_l = []
        for i in range(len(sentence_w)):    # Sentence
            w_list = []
            l_list = []
            for j in range(len(sentence_w[i])):  # Word
                w_list.append(self.toEmb(sentence_w[i][j]))
                if tagType is None:
                    l_list.append(sentence_l[i][j])
                else:
                    l_list.append(self.toTag(sentence_l[i][j], tagType))
            new_sentence_w.append(np.asarray(w_list))
            new_sentence_l.append(np.asarray(l_list))

        self.LOG(I + ' Words converted to Embedding : {} - Labels converted to vectorize: {}  -  {}  '.format(
            str(len(new_sentence_w)), str(tagType), str(len(new_sentence_l))))
        self.LOGResponse(True)
        return new_sentence_w, new_sentence_l

    # Convert Word to Embedding and Label to tagType : [None:TAG, False:t2i, True: tag vectorized]
    def convertedWordAndLabelForOnline(self, tagType=None):
        I = c.orange('*convertedWordAndLabel')
        sentence_w, sentence_l = self.getSentences()
        new_sentence_w = []
        new_sentence_l = []
        for i in range(len(sentence_w)):    # Sentence
            w_list = []
            l_list = []
            for j in range(len(sentence_w[i])):  # Word
                w_list.append(self.toEmb(sentence_w[i][j]))
                if tagType is None:
                    l_list.append(sentence_l[i][j])
                else:
                    l_list.append(self.toTag(sentence_l[i][j], tagType))
            new_sentence_w.append(np.asarray([w_list]))
            new_sentence_l.append(np.asarray([l_list]))

        self.LOG(I + ' Words converted to Embedding : {} - Labels converted to vectorize: {}  -  {}  '.format(
            str(len(new_sentence_w)), str(tagType), str(len(new_sentence_l))))
        self.LOGResponse(True)
        return new_sentence_w, new_sentence_l

    def toEmb(self, word):
        if word in self.w2i:
            return self.w2i[word]
        elif self.embedding is 'word2vec':
            return self.w2i['##']
        elif self.embedding is 'glove':
            return self.w2i['##']
        elif self.embedding is 'mine':
            return self.w2i['<UNK>']

    def toTag(self, tag, vectorized):
        if vectorized:
            TAG = len(self.i2t)
            vector = np.zeros((TAG,), dtype='int8')
            vector[self.t2i[tag]] = 1
            return vector
        else:
            return self.t2i[tag]

    def sentenceToEmb(self, sentence):
        return np.asarray([[self.toEmb(w) for w in sentence]])

    def addingPad(self, sentence, lenSequence):
        pad = lenSequence // 2
        context_word = []
        for i in range(pad):
            context_word.append(0)
        for i in range(len(sentence)):
            context_word.append(sentence[i])
        for i in range(pad):
            context_word.append(0)
        return np.asarray(context_word)

    def generateTestingXY(self):
        x, y = self.convertedWordAndLabel(tagType=None)
        return x, y

    def generateTrainingXY(self):
        x, y = self.convertedWordAndLabel(tagType=True)
        return x, y

    def generateForOnlineTrainingXY(self):
        x, y = self.convertedWordAndLabelForOnline(tagType=True)
        x, y = shuffle(x, y)
        return x, y

    def generateForOnlineTestingXY(self):
        x, y = self.convertedWordAndLabelForOnline(tagType=None)
        return x, y
