import numpy as np
import pandas as pd
import sklearn.cross_validation
import nltk.tokenize


def tags_to_array(tags):
    tags[pd.isnull(tags)] = ""
    tags += " "
    return np.sum(tags, axis=1)*3


def convert_feature(feature, X_train, target, X_test, test_index):
    feature_column = X_train[feature]

    if test_index is None:
        test_index = np.arange(X_test.shape[0])

    bincount = np.bincount(feature_column.astype(np.int64))
    bincount.resize(np.max(X_test[feature])+1)

    bincount_ones = np.bincount(feature_column[target == 1].astype(np.int64))
    bincount_ones.resize(np.max(X_test[feature])+1)

    X_test[feature+"_counts"].iloc[test_index] = 1.0*bincount[X_test[feature].iloc[test_index].values.astype(np.int64)]/feature_column.shape[0]
    X_test[feature+"_clicks"].iloc[test_index] = 1.0*bincount_ones[X_test[feature].iloc[test_index].values.astype(np.int64)]/feature_column.shape[0]


def categories_to_counters(X_train_, X_test_, y_train, folding=True, inplace=False):
    X_train = X_train_.copy(deep=True)
    X_test = X_test_.copy(deep=True)

    N_FOLDS = 25
    np.random.seed(1234)
    kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=N_FOLDS,
                                        shuffle=True, random_state=1234)

    X_train["OwnerUserId_counts"] = np.empty(X_train.shape[0])
    X_train["OwnerUserId_clicks"] = np.empty(X_train.shape[0])
    X_test["OwnerUserId_counts"] = np.empty(X_test.shape[0])
    X_test["OwnerUserId_clicks"] = np.empty(X_test.shape[0])


    feature = "OwnerUserId"
    for train_index, test_index in kf:
        convert_feature(feature, X_train.iloc[train_index],
                        y_train[train_index], X_train, test_index)
    convert_feature(feature, X_train, y_train, X_test, test_index=None)

    X_train[feature+"_clicks_by_counts"] = (X_train[feature+"_clicks"]*X_train.shape[0]*1.*(N_FOLDS-1)/N_FOLDS+1)/(X_train[feature+"_counts"]*X_train.shape[0]*1.*(N_FOLDS-1)/N_FOLDS+2)
    X_test[feature+"_clicks_by_counts"] = (X_test[feature+"_clicks"]*X_train.shape[0]+1)/(X_test[feature+"_counts"]*X_train.shape[0]+2)


    X_train.drop([feature], axis=1, inplace=True)
    X_test.drop([feature], axis=1, inplace=True)


    if inplace:
        X_train_ = X_train
        X_test_ = X_test

    return X_train, X_test


def extract_features(data):
    features = ["ReputationAtPostCreation",
                "OwnerUndeletedAnswerCountAtPostTime"]

    code_line_counter = np.zeros(len(data), dtype=np.int32)
    text_line_counter = np.zeros(len(data), dtype=np.int32)

    for i, text in enumerate(data["BodyMarkdown"]):

        lines = data["BodyMarkdown"][i].splitlines()

        for line in lines:
            if line.startswith("    "):
                code_line_counter[i] += 1
            else:
                text_line_counter[i] += 1

    X = data[features].copy(deep=True)
    X["CodeLineCounter"] = code_line_counter
    X["TextLineCounter"] = text_line_counter
    #X["LineCounter"] = code_line_counter + text_line_counter
    #X["CodeTextRatio"] = code_line_counter/(code_line_counter + text_line_counter)
    X["Date1"] = (data["PostCreationDate"].astype(np.int64)/1e6).astype(np.float32)
    X["Date2"] = (data["OwnerCreationDate"].astype(np.int64)/1e6).astype(np.float32)
    #X["PostId"] = data["PostId"]
    X["OwnerUserId"] = data["OwnerUserId"]
    X["TitleLength"] = data["Title"].apply(lambda x: len(x))
    X["TextLength"] = data["BodyMarkdown"].apply(lambda x: len(x))
    X["TitleNumWords"] = data["Title"].apply(lambda x: len(x.split()))
    X["TextNumWords"] = data["BodyMarkdown"].apply(lambda x: len(x.split()))
    X["SecondsDelta"] = ((data["PostCreationDate"] - data["OwnerCreationDate"])/np.timedelta64(1, 's')).astype(np.float64)

    #X[".SignCounter"] = data["BodyMarkdown"].apply(lambda x: x.count("."))
    #X["!SignCounter"] = data["BodyMarkdown"].apply(lambda x: x.count("!"))
    X["?SignCounter"] = data["BodyMarkdown"].apply(lambda x: x.count("?"))
    #X["@SignCounter"] = data["BodyMarkdown"].apply(lambda x: x.count("@"))
    X["urlCounter"] = data["BodyMarkdown"].apply(lambda x: x.count("http://") + x.count("https://") + x.count(".com") + x.count(".org"))
    X["thanksCounter"] = data["BodyMarkdown"].apply(lambda x: x.lower().count("thank"))
    X["modifiedCounter"] = data["BodyMarkdown"].apply(lambda x: x.lower().count("last_modified"))
    X["numSentences"] = data["BodyMarkdown"].apply(lambda x: len(nltk.tokenize.sent_tokenize(x)))
    #X["meanSentLength"] = data["BodyMarkdown"].apply(lambda x: nltk.tokenize.sent_tokenize(x)).apply(lambda x: list(map(lambda y: nltk.tokenize.word_tokenize(y), x))).apply(lambda x: list(map(lambda y: len(y), x))).apply(lambda x: np.mean(x))
    #Длина предложения - средняя, максимальная, минимальная, стд
    #X["learn"] = data["BodyMarkdown"].apply(lambda x: x.lower().count("teach"))
    #X["paper"] = data["BodyMarkdown"].apply(lambda x: x.lower().count("paper"))
    #X["IsQuestion"] = (X["?SignCounter"] == 0).astype(np.int32)

    #X["hour"] = data["PostCreationDate"].apply(lambda x: x.hour)
    #X["DaysDelta"] = ((data["PostCreationDate"] - data["OwnerCreationDate"])/np.timedelta64(1, 'D')).astype(np.int32)
    #X["hour"] = data["PostCreationDate"].apply(lambda x: x.hour)
    #X["DaysDeltaZero"] = X["DaysDelta"].apply(lambda x: np.abs(x) < 0.1)
    #X["DaysDeltaSamller1"] = X["DaysDelta"].apply(lambda x: x < 1)
    #X["ReputationAtPostCreationSmaller1"] = X["ReputationAtPostCreation"].apply(lambda x: x < 1)
    #print(data.columns)
    #X["OwnerUserId_clicks_by_countsEqual05"] = X["OwnerUserId_clicks_by_counts"].apply(lambda x: np.abs(x - 0.5) < 0.001)

    return X


def transform_features(X_train_, X_test_):
    features = ["ReputationAtPostCreation",
                "OwnerUndeletedAnswerCountAtPostTime",
                "TitleLength", "TextLength", "TitleNumWords", "TextNumWords",
                "SecondsDelta"]#, "OwnerUserId_counts"]

    X_train = X_train_.copy(deep=True)
    X_test = X_test_.copy(deep=True)

    for feature in features:
        X_train.ix[X_train[feature] < 0, feature] = 0
        X_test.ix[X_test[feature] < 0, feature] = 0
        X_train[feature] = np.log(X_train[feature]+1)
        X_test[feature] = np.log(X_test[feature]+1)

    return X_train, X_test

