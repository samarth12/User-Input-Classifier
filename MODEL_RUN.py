import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import features
#clf = RandomForestClassifier(n_jobs=2, n_estimators = 100)

user_input = "Is the earth flat"

def classify_sentence(clf,user_input):
    keys = ["id",
    "wordCount",
    "stemmedCount",
    "stemmedEndNN",
    "CD",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PRP",
    "VBG",
    "VBZ",
    "startTuple0",
    "endTuple0",
    "endTuple1",
    "endTuple2",
    "verbBeforeNoun",
    "qMark",
    "qVerbCombo",
    "qTripleScore",
    "sTripleScore",
    "class"]
    myFeatures = features.features_dict('1',user_input, 'X')
    values=[]
    for key in keys:
        values.append(myFeatures[key])
    s = pd.Series(values)
    width = len(s)
    myFeatures = s[1:width-1]
    #clf.fit(train[features], train['class'])
    predict = clf.predict([myFeatures])
    #predout = pd.DataFrame({ 'id' : '1', 'predicted' : predict, 'actual' : test['class'] })
    #print (predout)
    print(predict[0].strip())

def classify_model():
    FNAME = 'data/features_extracted.csv'
    df = pd.read_csv(filepath_or_buffer = FNAME, )
    df.columns = df.columns[:].str.strip()
    df['class'] = df['class'].map(lambda x: x.strip())
    width = df.shape[1]
    #split into test and training
    np.random.seed(seed=1)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]
    features = df.columns[1:width-1]
    clf = RandomForestClassifier(n_jobs=2, n_estimators = 100)
    final = clf.fit(train[features], train['class'])
    # Predict against test set
    preds = final.predict(test[features])
    predout = pd.DataFrame({ 'id' : test['id'], 'predicted' : preds, 'actual' : test['class'] })
    classify_sentence(clf,user_input)
    #print (predout)
    return clf

classify_model()
