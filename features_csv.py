import csv
import sys
import hashlib
import features

if len(sys.argv) > 1:
    FNAME = sys.argv[1]
else:
    FNAME = './data/training_data.csv'
print("reading input from ", FNAME)


if len(sys.argv) > 2:
    FOUT = sys.argv[2]
else:
    FOUT = './data/features_extracted.csv'
print("Writing output to ", FOUT)

fin = open(FNAME, 'rt')
fout = open(FOUT, 'wt', newline='')

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

reader = csv.reader(fin)

loopCount = 0
next(reader)
for line in reader:
    sentence = line[0]
    c = line[1]
    id = hashlib.md5(str(sentence).encode('utf-8')).hexdigest()[:16]

    output = ""
    header = ""


    f = features.features_dict(id,sentence, c)

    for key in keys:
        value = f[key]
        header = header + ", " + key
        output = output + ", " + str(value)

    if loopCount == 0:
        header = header[1:]
        print(header)
        fout.writelines(header + '\n')

    output = output[1:]

    loopCount = loopCount + 1
    print(output)
    fout.writelines(output + '\n')


fin.close()
fout.close()
