
import nltk
from nltk import word_tokenize

lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')
from nltk.corpus import stopwords

import pandas as pd

import csv
import sys
import hashlib
import re
import string
import itertools

line = ["xxx","Hello my name is Samarth Bhandari, how are you?",0,"S"]

pos = []

output = ""
header = ""

VerbCombos = ['VB',
              'VBD',
              'VBG',
              'VBN',
              'VBP',
              'VBZ',
              'WDT',
              'WP',
              'WP$',
              'WRB',
              'MD']

questionTriples = ['CD-VB-VBN',
                   'MD-PRP-VB' ,
                   'MD-VB-CD' ,
                   'NN-IN-DT' ,
                   'PRP-VB-PRP' ,
                   'PRP-WP-NNP' ,
                   'VB-CD-VB' ,
                   'VB-PRP-WP' ,
                   'VBZ-DT-NN' ,
                   'WP-VBZ-DT' ,
                   'WP-VBZ-NNP' ,
                   'WRB-MD-VB']

statementTriples = ['DT-JJ-NN',
                   'DT-NN-VBZ',
                   'DT-NNP-NNP',
                   'IN-DT-NN',
                   'IN-NN-NNS',
                   'MD-VB-VBN',
                   'NNP-IN-NNP',
                   'NNP-NNP-NNP',
                   'NNP-VBZ-DT',
                   'NNP-VBZ-NNP',
                   'NNS-IN-DT',
                   'VB-VBN-IN',
                   'VBZ-DT-JJ']


startTuples = ['NNS-DT',
'WP-VBZ',
               'WRB-MD']

endTuples = ['IN-NN',
             'VB-VBN',
             'VBZ-NNP']

feature_keys = ["id",
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


def strip_sentence(sentence):
    sentence = sentence.strip(",")
    sentence = ''.join(filter(lambda x: x in string.printable, sentence))
    sentence = sentence.translate(str.maketrans('','',string.punctuation))
    return(sentence)

def exists_pair_combos(comboCheckList, sentence):
    pos = get_pos(sentence)
    tag_string = "-".join([ i[1] for i in pos ])
    combo_list = []

    for pair in itertools.permutations(comboCheckList,2):
        if(pair[0] == "MD"):
            pair = ["",""]
        combo_list.append("-".join(pair))

    if any(code in tag_string for code in combo_list):
	    return 1
    else:
        return 0


def get_pos(sentence):
    sentenceParsed = word_tokenize(sentence)
    return(nltk.pos_tag(sentenceParsed))


def count_qmark(sentence):
    return(sentence.count("?") )

def count_POSType(pos, ptype):
    count = 0
    tags = [ i[1] for i in pos ]
    return(tags.count(ptype))

def exists_vb_before_nn(pos):
    pos_tags = [ i[1] for i in pos ]
    pos_tags = [ re.sub(r'V.*','V', str) for str in pos_tags ]
    pos_tags = [ re.sub(r'NN.*','NN', str) for str in pos_tags ]

    vi =99
    ni =99
    mi =99


    if "NN" in pos_tags:
        ni = pos_tags.index("NN")

    if "V" in pos_tags:
        vi = pos_tags.index("V")

    if "MD" in pos_tags:
        mi = pos_tags.index("MD")

    if vi < ni or mi < ni :
        return(1)
    else:
        return(0)


def exists_stemmed_end_NN(stemmed):
    stemmedEndNN = 0
    stemmed_end = get_first_last_tuples(" ".join(stemmed))[1]
    if stemmed_end == "NN-NN":
        stemmedEndNN = 1
    return(stemmedEndNN)


def exists_startTuple(startTuple):
    exists_startTuples = []
    for tstring in startTuples:
        if startTuple in tstring:
            exists_startTuples.append(1)
        else:
            exists_startTuples.append(0)
        return(exists_startTuples)

def exists_endTuple(endTuple):
    exists_endTuples = []
    for tstring in endTuples:
        if endTuple in tstring:
            exists_endTuples.append(1)
        else:
            exists_endTuples.append(0)
    return(exists_endTuples)

def exists_triples(triples, tripleSet):
    exists = []
    for tstring in tripleSet:
        if tstring in triples:
            exists.append(1)
        else:
            exists.append(0)
    return(exists)

def get_triples(pos):
    list_of_triple_strings = []
    pos = [ i[1] for i in pos ]
    n = len(pos)

    if n > 2:
        for i in range(0,n-2):
            t = "-".join(pos[i:i+3])
            list_of_triple_strings.append(t)
    return list_of_triple_strings

def get_first_last_tuples(sentence):
    first_last_tuples = []
    sentenceParsed = word_tokenize(sentence)
    pos = nltk.pos_tag(sentenceParsed)
    pos = [ i[1] for i in pos ]

    n = len(pos)
    first = ""
    last = ""

    if n > 1:
        first = "-".join(pos[0:2])
        last = "-".join(pos[-2:])

    first_last_tuples = [first, last]
    return first_last_tuples

def lemmatize(sentence):

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)

    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w.lower())
    lem = []
    for w in filtered_sentence:
        lem.append(lemma.lemmatize(w))

    return lem

def stematize(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)

    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    stemmed = []
    for w in filtered_sentence:
        stemmed.append(sno.stem(w))

    return stemmed


def get_string(id,sentence,c="X"):
    header,output = "",""
    pos = get_pos(sentence)

    qMark = count_qmark(sentence)
    sentence = strip_sentence(sentence)
    #lemmed = lemmatize(sentence)
    stemmed = stematize(sentence)
    wordCount = len(sentence.split())
    stemmedCount = len(stemmed)

    qVerbCombo = exists_pair_combos(VerbCombos,sentence)

    verbBeforeNoun = exists_vb_before_nn(pos)

    output = id + ","  + str(wordCount) + "," + str(stemmedCount) + "," + str(qVerbCombo)+ "," + str(qMark) + "," + str(verbBeforeNoun)
    header = header + "id,wordCount,stemmedCount,qVerbCombo,qMark,verbBeforeNoun"


    for ptype in ["VBG", "VBZ", "NNP", "NN", "NNS", "NNPS","PRP", "CD"]:
        output = output + "," + str( count_POSType(pos,ptype) )
        header = header + "," + ptype

    output = output + "," + str(exists_stemmed_end_NN(stemmed))
    header = header + ",StemmedEndNN,"


    startTuple,endTuple = get_first_last_tuples(sentence)

    l = exists_startTuple(startTuple)
    output = output + "," + ",".join(str(i) for i in l)
    for i in range(0,len(l)):
        header = header + "startTuple" + str(i+1) + ","

    l = exists_endTuple(endTuple)
    output = output + "," + ",".join(str(i) for i in l)
    for i in range(0,len(l)):
        header = header + "endTuple" + str(i+1) + ","


    triples = get_triples(pos)

    l = exists_triples(triples, questionTriples)
    total = sum(l)
    output = output + "," + str(total)
    header = header + "qTripleScore" + ","

    l = exists_triples(triples, statementTriples)
    total = sum(l)
    output = output + "," + str(total)
    header = header + "sTripleScore" + ","

    output = output + "," + c
    header = header + "class"

    return output,header

def features_dict(id,sentence,c="X"):
    features = {}
    pos = get_pos(sentence)

    features["id"] = id
    features["qMark"] = count_qmark(sentence)
    sentence = strip_sentence(sentence)
    stemmed = stematize(sentence)
    startTuple,endTuple = get_first_last_tuples(sentence)

    features["wordCount"] = len(sentence.split())
    features["stemmedCount"] = len(stemmed)
    features["qVerbCombo"] = exists_pair_combos(VerbCombos,sentence)
    features["verbBeforeNoun"] = exists_vb_before_nn(pos)

    for ptype in ["VBG", "VBZ", "NNP", "NN", "NNS", "NNPS","PRP", "CD"]:
        features[ptype] = count_POSType(pos,ptype)

    features["stemmedEndNN"] = exists_stemmed_end_NN(stemmed)

    l = exists_startTuple(startTuple)
    for i in range(0,len(l)):
        features["startTuple" + str(i)] = l[i]

    l = exists_endTuple(endTuple)
    for i in range(0,len(l)):
        features["endTuple" + str(i)] = l[i]


    triples = get_triples(pos)

    l = exists_triples(triples, questionTriples)
    features["qTripleScore"] = sum(l)

    l = exists_triples(triples, statementTriples)
    features["sTripleScore"] = sum(l)

    features["class"] = c

    return features


def features_series(features_dict):
    values=[]
    for key in feature_keys:
        values.append(features_dict[key])

    features_series = pd.Series(values)

    return features_series

## MAIN ##
if __name__ == '__main__':


    print("Starting...")

    c = "X"
    header = ""
    output = ""

    if len(sys.argv) > 1:
        sentence = sys.argv[1]
    else:
        sentence = line[1]

    id = hashlib.md5(str(sentence).encode('utf-8')).hexdigest()[:16]

    features = features_dict(id,sentence, c)
    pos = get_pos(sentence)
    print(pos)

    print(features)
    for key,value in features.items():
        print(key, value)

    #header string
    for key, value in features.items():
       header = header + ", " + key
       output = output + ", " + str(value)
    header = header[1:]
    output = output[1:]
    print("HEADER:", header)
    print("VALUES:", output)
