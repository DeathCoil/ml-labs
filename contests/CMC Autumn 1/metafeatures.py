import numpy as np
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.ensemble

def linear_model_as_feature(X_train, y_train, X_test, load=True):

    feature_train = None
    feature_test = None

    if load:
        feature_train = np.load("metafeatures/linear_train")
        feature_test = np.load("metafeatures/linear_test")
    else:
        feature_train = np.zeros(X_train.shape[0])
        feature_test = np.zeros(X_test.shape[0])
        model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.6)
        kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=25,
                                            shuffle=True, random_state=1234)

        for train_index, test_index in kf:
            model.fit(X_train[train_index], y_train[train_index])
            feature_train[test_index] = model.predict_proba(X_train[test_index])[:, 1]

        model.fit(X_train, y_train)
        feature_test = model.predict_proba(X_test)[:, 1]

        feature_train.dump("metafeatures/linear_train")
        feature_test.dump("metafeatures/linear_test")

    return feature_train, feature_test


def w2v_model_as_feature(X_train, y_train, X_test, load=True, model_to_train="rf"):

    feature_train = None
    feature_test = None
    if load:
        if model_to_train == "rf":
            feature_train = np.load("metafeatures/w2v_train_rf_5folds")
            feature_test = np.load("metafeatures/w2v_test_rf_5folds")
        elif model_to_train == "linear":
            feature_train = np.load("metafeatures/w2v_train_linear_25folds")
            feature_test = np.load("metafeatures/w2v_test_linear_25folds")
    else:
        feature_train = np.zeros(X_train.shape[0])
        feature_test = np.zeros(X_test.shape[0])
        model = None
        kf = None
        if model_to_train == "rf":
            model = sklearn.ensemble.RandomForestClassifier(criterion="gini", max_depth=16, n_estimators=300,
                                                            min_samples_leaf=8, n_jobs=4, random_state=1234)
            kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=5,
                                                shuffle=True, random_state=1234)
        elif model_to_train == "linear":
            model = sklearn.linear_model.LogisticRegression(C=90, penalty="l2")
            kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=25,
                                                shuffle=True, random_state=1234)

        for train_index, test_index in kf:
            model.fit(X_train[train_index], y_train[train_index])
            feature_train[test_index] = model.predict_proba(X_train[test_index])[:, 1]

        model.fit(X_train, y_train)
        feature_test = model.predict_proba(X_test)[:, 1]
        if model_to_train == "rf":
            feature_train.dump("metafeatures/w2v_train_rf_5folds")
            feature_test.dump("metafeatures/w2v_test_rf_5folds")
        elif model_to_train == "linear":
            feature_train.dump("metafeatures/w2v_train_linear_25folds")
            feature_test.dump("metafeatures/w2v_test_linear_25folds")

    return feature_train, feature_test