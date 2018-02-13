import numpy as np
import os
import pickle
import gensim


class c:
    @staticmethod
    def red(string):
        if string is None:
            string = 'None'
        return '\033[31m' + string + '\033[0m'

    @staticmethod
    def green(string):
        if string is None:
            string = 'None'
        return '\033[32m' + string + '\033[0m'

    @staticmethod
    def orange(string):
        if string is None:
            string = 'None'
        return '\033[33m' + string + '\033[0m'

    @staticmethod
    def blue(string):
        if string is None:
            string = 'None'
        return '\033[34m' + string + '\033[0m'

    @staticmethod
    def purple(string):
        if string is None:
            string = 'None'
        return '\033[35m' + string + '\033[0m'


DH = c.blue('[Data-Helper]')


# Create the weights numpy array and the word2index dictionary file
def create_word2vec_data(weights=True, word2index=True):
    I = DH + c.blue('(create_word2vec_data) ')
    print(I + ' Weight: {} - word2index: {}'.format(weights, word2index))
    w2v_name = 'GoogleNews-vectors-negative300.bin.gz'
    weights_name = 'syn0.npz'
    w2i_name = 'word2index.data'

    resource_path = os.path.join(os.getcwd(), 'resources')
    w2v_dir_path = os.path.join(resource_path, 'word2vec')
    w2v_path = os.path.join(w2v_dir_path, w2v_name)
    weights_path = os.path.join(w2v_dir_path, weights_name)
    w2i_path = os.path.join(w2v_dir_path, w2i_name)

    just_created_weights = os.path.exists(weights_path)
    just_created_w2i = os.path.exists(w2i_path)
    create = True
    if just_created_weights and just_created_w2i:
        create = False

    # Has be created
    if create:
        print(I + ' Data as to be created.')
        save_word2vec_weights(w2v_path, weights_path, w2i_path)
        print(I + ' Data for using word2vec - Created! ')

    syn0 = None
    w2i = None

    if weights and word2index:
        syn0, w2i = load_word2vec_data(weights_path, w2i_path)
    elif weights:
        syn0, w2i = load_word2vec_data(weights_path, None)
    elif word2index:
        syn0, w2i = load_word2vec_data(None, w2i_path)
    return syn0, w2i


# Load the weights numpy array and the word2index dictionary file
def load_word2vec_data(weights_path, word2index_path):
    I = DH + c.blue('(load_word2vec_data) ')
    weights = None
    w2i = None
    if weights_path:
        weights = np.load(weights_path)['syn0']
        print(I + ' Weigths Loaded! --> ' + weights_path)
    if word2index_path:
        with open(word2index_path, 'rb') as f:
            w2i = pickle.loads(f.read())
        print(I + ' Word2index Loaded! --> ' + word2index_path)
    return weights, w2i


# Save the weights numpy array and the word2index dictionary file
def save_word2vec_weights(word2vec_negative_sample_path, weights_path, word2index_path):
    I = DH + c.green('(save_word2vec_weights) ')
    print(I + ' Creation data for word2vec! ')
    print(I + ' Try to load the gensim model of word2vec.')
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_negative_sample_path, binary=True)
    print(I + ' Model word2vec Loaded!')

    weights = model.syn0
    with open(weights_path, 'wb') as npz_file:
        np.savez(npz_file, syn0=weights)
    print(I + ' Weights syn0 saved to --> ' + weights_path)

    vocab = dict([(k, v.index + 1) for k, v in model.vocab.items()])
    with open(word2index_path, 'wb') as w2i_file:
        pickle.dump(vocab, w2i_file)
    print(I + ' Word2index saved to --> ' + word2index_path)


def load_glove_as_dictionary():
    I = DH + c.green('(load_glove_as_dictionary) ')
    GLOVE_DIR = '/Users/marcotreglia/Dropbox/NLP/homework_2/glove'
    GLOVE_FILE = 'glove.6B.100d.txt'
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector
    f.close()

    print(I + 'Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def universal_pos_tags():
    tags = dict(
        ADJ='adjective',
        ADP='adposition',
        ADV='adverb',
        AUX='auxiliary',
        CONJ='coordinating conjunction',
        DET='determiner',
        INTJ='interjection',
        NOUN='noun',
        NUM='numeral',
        PART='particle',
        PRON='pronoun',
        PROPN='proper noun',
        PUNCT='punctuation',
        SCONJ='subordinating conjunction',
        SYM='symbol',
        VERB='verb',
        X='other')

    return tags


def get_tag_mapping():
    tags = universal_pos_tags()
    tags_to_index = dict((tag, idx) for idx, tag in enumerate(tags))
    index_to_tags = dict((idx, tag) for idx, tag in enumerate(tags))
    return tags_to_index, index_to_tags


# Return the index with the best probability
def index_best_probability(preds, temperature=1.0, softmax=False):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    if softmax:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        best_index = np.argmax(probas)
    else:
        best_index = np.argmax(preds)
    return best_index


# Convert the prediction to Tag
def sentence_prob_to_tag(sentence_predicted_probability, i2t):
    tags = []
    for prob_tags in sentence_predicted_probability:
        best_prob_idx = index_best_probability(prob_tags)
        tags.append(i2t[best_prob_idx])
    return np.asarray(tags)
