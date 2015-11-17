import numpy as np
import gensim.models.word2vec
import nltk.tokenize
import logging

def make_word2vec_model(text_train_, text_test_):
    text_train = text_train_.copy()
    text_test = text_test_.copy()
    texts = np.concatenate((text_train, text_test))

    sentences = []
    for text in texts:
        sentences += nltk.tokenize.sent_tokenize(text)

    sentences = list(map(lambda x: nltk.tokenize.word_tokenize(x), sentences))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)


    model = gensim.models.word2vec.Word2Vec(sentences, workers=12, size=150,
                                            min_count=10, window=10,
                                            sample=0.0001, seed=1234)
    model.init_sims(replace=True)
    model.save("w2v/basic_word2vec_model")

    return model


def texts_to_words(data, load_model=True):
    model = None
    if load_model:
        model = gensim.models.word2vec.Word2Vec.load("w2v/basic_word2vec_model")
    else:
        raise("Error")

    w2v_words = set(model.index2word)
    words = list(map(lambda x: nltk.tokenize.word_tokenize(x), data))

    feature = np.zeros((len(words), 150))

    for i, words in enumerate(words):
        n = 0
        for word in words:
            if word in w2v_words:
                feature[i] += model[word]
                n += 1
        if n:
            feature[i] /= n

    return feature


def word2vec_features(text_train, text_test, load=True):
    if load:
        feature_train = np.load("w2v/word2vec_feature_train")
        feature_test = np.load("w2v/word2vec_feature_test")
        return feature_train, feature_test
    else:
        feature_train = texts_to_words(text_train, load_model=True)
        feature_test = texts_to_words(text_test, load_model=True)
        feature_train.dump("w2v/word2vec_feature_train")
        feature_test.dump("w2v/word2vec_feature_test")
        return feature_train, feature_test