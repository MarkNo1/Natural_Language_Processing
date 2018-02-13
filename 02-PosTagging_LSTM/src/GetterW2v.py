from data_helper import create_word2vec_data
from KerasManager import KerasManager
from Conllu_Manager import Conllu_Manager
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import numpy as np
import os


syn0, _ = create_word2vec_data(word2index=False)
Km = KerasManager()


# Data
dataDir = os.path.join(os.getcwd(), 'data')
trainPath = os.path.join(dataDir, 'en-ud-train.conllu')
train = Conllu_Manager('train', dataSetPath=trainPath, embedding='word2vec')
x, y = train.generateForOnlineTrainingXY()


x_ravel = []
for i in range(len(x)):
    for j in range(x[i][0].shape[0]):
        word = x[i][0][j]
        x_ravel.append(word)
x_ravel = np.asarray(x_ravel)

# All the word appear in the Training File
words_single_appear = sorted(list(set(x_ravel)))
words_single_appear = np.asarray(words_single_appear)

# Load the W2v model in Keras
Km.Extrapoling_Word2vec_words(syn0)

# Extrac all the vector representation of the word in train vocabulary
# Notice:
#           I tested also doing the pca reduction on the  whole vacobulary
#           getting the same result of explained_variance_ratio . So what i
#           realize is that pca don't need more n_samples to achive the
#           varianche between the components. :)
#

max_vocab = len(words_single_appear)
wordsToMatrix = []
for i in tqdm(range(max_vocab)):
    wordsToMatrix.append(Km.model.predict(words_single_appear[i:i + 1]))


# Reshape the words matrix to [n_sample, feature]
wordsToMatrix = np.asarray(wordsToMatrix)
wordsMatrix = wordsToMatrix.reshape((17408, 300))

# Dimesionality reduction to [reduc_dimension]
reduc_dimension = 20
pca = IncrementalPCA(n_components=reduc_dimension)
w_reduced = pca.fit_transform(wordsMatrix)

with open('TrainingWord-' + str(reduc_dimension) + 'D.csv', 'w') as f:
    for i in range(max_vocab):
        f.write(str(words_single_appear[i]) + ',' + ','.join(map(str, w_reduced[i])) + '\n')

# w2v = dict()
# for i in range(len(words_single_appear) - 1):
#     w2v.update({words_single_appear[i]: wordsMatrix[i]})
