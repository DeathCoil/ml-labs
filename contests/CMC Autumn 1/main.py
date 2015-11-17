import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text
import sklearn.grid_search
import sklearn.linear_model
import sklearn.ensemble

import input_output as io
import feature_extraction as fe
import word2vec_model as w2v
import metafeatures as mf
import tuning as tuning



def watch_errors(X_train, y_train):
    preds = np.zeros(y_train.shape[0], dtype=np.float32)
    model = sklearn.ensemble.RandomForestClassifier(max_depth=12, n_estimators=500,
                                                    min_samples_leaf=8, min_samples_split=2)

    kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=5,
                                        shuffle=True, random_state=1234)

    for train_index, test_index in kf:
        model.fit(X_train[train_index], y_train[train_index])
        preds[test_index] = model.predict_proba(X_train[test_index])[:, 1]

    np.random.seed(1234)
    error_index = np.random.choice(np.arange(X_train.shape[0])[preds != y_train], 25)
    print(error_index)

    return preds


def make_predictions(model, X_train, target, X_test):
    model.fit(X_train, target)
    return model.predict_proba(X_test)[:, 1]


def plot_data(X_train, target, X_test):
    for feature in X_train.columns:
        plt.figure()
        plt.hist(X_train[feature][target == 1].values, 100, alpha=0.5, normed=1, facecolor='g', label="1")
        plt.hist(X_train[feature][target == 0].values, 100, alpha=0.5, normed=1, facecolor='b', label="0")
        plt.legend(loc="upper right")
        plt.title(feature)
        plt.grid(True)
        plt.show()


def ensemble(target):
    preds1 = np.load("ensemble_results/rf_stacked")
    preds2 = np.load("ensemble_results/linear_model")
    scores = np.zeros(101, dtype=np.float32)
    for alpha in range(0, 101):
        scores[alpha] = sklearn.metrics.roc_auc_score(target, 0.01*alpha*preds1 + np.max(1 - 0.01*alpha, 0)*preds2)
    print(np.max(scores), np.unravel_index(scores.argmax(), scores.shape), scores[0], scores[100])


def get_tfidf(train, test):
    tags_columns = ["Tag1", "Tag2", "Tag3", "Tag4", "Tag5"]
    text_train = train["Title"].values + " " + train["BodyMarkdown"].values + " " + fe.tags_to_array(train[tags_columns].values)
    text_test = test["Title"].values + " " + test["BodyMarkdown"].values + " " + fe.tags_to_array(test[tags_columns].values)

    tfidf1 = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                                                             min_df=25)
    text_train_tfidf = tfidf1.fit_transform(text_train)
    text_test_tfidf = tfidf1.transform(text_test)

    text_train = train["Title"].values
    text_test = test["Title"].values

    tfidf2 = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                                                             min_df=5)
    text_train_tfidf = scipy.sparse.hstack((text_train_tfidf, tfidf2.fit_transform(text_train)), format="csr")
    text_test_tfidf = scipy.sparse.hstack((text_test_tfidf, tfidf2.transform(text_test)), format="csr")

    return text_train_tfidf, text_test_tfidf


def rf_model(train, target, test, text_train_tfidf, text_test_tfidf):

    text_train = train["Title"].values + ". " + train["BodyMarkdown"].values
    text_test = test["Title"].values + ". " + test["BodyMarkdown"].values
    print("Creating word2vec model...")
    w2v.make_word2vec_model(text_train, text_test)
    wv_train, wv_test = w2v.word2vec_features(text_train, text_test, load=False)

    X_train, X_test = fe.extract_features(train), fe.extract_features(test)


    X_train, X_test = fe.categories_to_counters(X_train, X_test, target)
    X_train, X_test = fe.transform_features(X_train, X_test)
    print("Creating linear model metafeature...")
    X_train["LinearModelText"], X_test["LinearModelText"] = mf.linear_model_as_feature(text_train_tfidf, target, text_test_tfidf, load=False)
    print("Creating word2vec model metafeature...")
    X_train["w2vModelRFText"], X_test["w2vModelRFText"] = mf.w2v_model_as_feature(wv_train, target, wv_test, load=False, model_to_train="rf")

    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = sklearn.ensemble.RandomForestClassifier(criterion="entropy", max_depth=14, n_estimators=2000,
                                                    min_samples_leaf=4, min_samples_split=16, n_jobs=4, random_state=1234)

    result = make_predictions(model, X_train, target, X_test)
    io.save_result(test["PostId"], result)

    return result

def linear_model(train, target, test, text_train_tfidf, text_test_tfidf):

    X_train, X_test = fe.extract_features(train), fe.extract_features(test)

    X_train, X_test = fe.categories_to_counters(X_train, X_test, target)
    X_train, X_test = fe.transform_features(X_train, X_test)

    feature_train = np.load("w2v/word2vec_feature_train")
    feature_test = np.load("w2v/word2vec_feature_test")

    X_train = np.column_stack((X_train.values, feature_train))
    X_test = np.column_stack((X_test.values, feature_test))


    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    X_train = scipy.sparse.hstack((text_train_tfidf, X_train), format="csr")
    X_test = scipy.sparse.hstack((text_test_tfidf, X_test), format="csr")

    model = sklearn.linear_model.LogisticRegression(C=0.7, penalty="l2")

    result = make_predictions(model, X_train, target, X_test)
    io.save_result(test["PostId"], result)

    return result



train, target, test = io.load_data()
text_train_tfidf, text_test_tfidf = get_tfidf(train, test)
preds1 = rf_model(train, target, test, text_train_tfidf,
                  text_test_tfidf)
preds2 = linear_model(train, target, test, text_train_tfidf,
                      text_test_tfidf)

result = 0.7*preds1 + 0.3*preds2
io.save_result(test["PostId"], result)
