import numpy as np
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics

def ensemble_tuning(X_train, y_train):
    kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=5,
                                        shuffle=True, random_state=1234)

    model1 = sklearn.ensemble.RandomForestClassifier(criterion="entropy", max_depth=15, n_estimators=500,
                                                    min_samples_leaf=4, min_samples_split=16, random_state=1234)
    #model2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=170, algorithm="kd_tree", weights="distance")
    model2 = sklearn.linear_model.LogisticRegression(C=2, penalty="l2")

    scores = np.zeros((5, 101), dtype=np.float32)

    for fold, (train_index, test_index) in enumerate(kf):
        model1.fit(X_train[train_index], y_train[train_index])
        pred1 = model1.predict_proba(X_train[test_index])[:, 1]

        model2.fit(X_train[train_index], y_train[train_index])
        pred2 = model2.predict_proba(X_train[test_index])[:, 1]

        print("Calculating scores...")
        for alpha in np.ndindex(101):
            scores[fold][alpha] = sklearn.metrics.roc_auc_score(y_train[test_index], 0.01*alpha[0]*pred1 + np.max(1 - 0.01*alpha[0], 0)*pred2)

        sc1 = np.mean(scores, axis = 0) * 1.0 / (fold+1) * 5
        print(np.max(sc1), np.unravel_index(sc1.argmax(), sc1.shape), sc1[0], sc1[100])

    scores1 = np.mean(scores, axis = 0)
    print(np.max(scores1), np.unravel_index(scores1.argmax(), scores1.shape), scores1[0], scores1[100])


def parametr_tuning(X_train, y_train):
    #linear model on tf-idf only
    #model = sklearn.linear_model.LogisticRegression()
    #param_grid = {"C" : [0.6], "penalty" : ["l2"]}

    #min_samples_split=32 ?
    #model = sklearn.ensemble.RandomForestClassifier(criterion="gini", max_depth=18,
    #                                                n_jobs=3, min_sample_leaf=4, random_state=1234)
    #param_grid = {"n_estimators" : [300]}
    #model = sklearn.ensemble.RandomForestClassifier(criterion="gini", max_depth=16, n_jobs=1,
    #                                                min_samples_leaf=8, random_state=1234)
    #param_grid = {"n_estimators" : [50]}
    #param_grid = {"n_neighbors" : [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], "metric" : ["euclidean", "manhattan", "cosine"]}
    #model = sklearn.neighbors.KNeighborsClassifier(algorithm="kd_tree", n_neighbors=170)

    model = sklearn.linear_model.LogisticRegression()
    param_grid = {"C" : [0.7], "penalty" : ["l2"]}

    cv = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=5,
                                        shuffle=True, random_state=1234)

    gs = sklearn.grid_search.GridSearchCV(model, param_grid, scoring="roc_auc",
                                          cv=cv, verbose=10, n_jobs=1)
    gs.fit(X_train, y_train)

    print("Best score is: ", gs.best_score_)
    print("Best parametrs:")

    best_params = gs.best_estimator_.get_params()

    for param_name in sorted(best_params.keys()):
        print(param_name, ":", best_params[param_name])