import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import features
from sklearn import metrics
import sys
#clf = RandomForestClassifier(n_jobs=2, n_estimators = 100)

user_input = str(sys.argv)

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
    FNAME = '/analysis/featuresDump.csv'
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
    dictlist =[]

    for key, value in test['class'].iteritems():
        temp = [key,value]
        dictlist.append(temp)
    true_class = [ row[1] for row in dictlist]
    p = metrics.precision_score(true_class, preds, average='macro')
    r = metrics.recall_score(true_class, preds, average='micro')
    f1 = metrics.fbeta_score(true_class, preds, average='macro', beta=0.5)
    print(p)
    print(r)
    print(f1)
    return clf

classify_model()
