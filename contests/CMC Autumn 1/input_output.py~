import pandas as pd
import numpy as np
import datetime as datetime
import sklearn.cross_validation


def load_data():
    train = pd.read_csv("input/train-contest.csv",
                        parse_dates=["PostCreationDate", "OwnerCreationDate"])
    test = pd.read_csv("input/test-contest-second.csv",
                       parse_dates=["PostCreationDate", "OwnerCreationDate"])
    target = train["OpenStatus"].values
    train.drop("OpenStatus", axis=1, inplace=True)

    return train, target, test


def save_result(index, result):
    df = pd.DataFrame({"PostId": index, "OpenStatus": result})
    df.to_csv("output/" + str(datetime.datetime.now()), index=False)


def dump_train_results(X_train, target, model, name, n_folds=5):
    kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=n_folds,
                                        shuffle=True, random_state=1234)
    preds = np.zeros(target.shape[0])

    for train_index, test_index in kf:
        model.fit(X_train[train_index], target[train_index])
        preds[test_index] = model.predict_proba(X_train[test_index])[:, 1]

    preds.dump(name)
