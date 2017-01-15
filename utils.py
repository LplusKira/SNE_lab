import sys
import traceback
from config import USR_LABELS_FIELDS

# assign prepared the usr-labels file to each usr
# input file format (per line):
#    usrid|one-hot encoded feature_1|...|one-hot encoded feature_n
def get_labels(usr2labels_file):
    usr2labels = {}
    fd = open(usr2labels_file, 'r')
    for line in fd:
        try:
            line = line.strip().split('|')
            usr = int(line[0])
            tmplabels = []
            for ind in USR_LABELS_FIELDS:
                lablels = map(lambda v: int(v), line[ind].split(',')) if len(line[ind]) > 0 else []
                tmplabels.append(lablels)
            usr2labels[usr] = tmplabels
        except: 
            print traceback.format_exc()
            raise
    fd.close()
    return usr2labels

def load_data(usr2labels, usr2representation):
    X_train = []
    y_train = []
    for usr in usr2representation:
        try:
            X_train.append(usr2representation[usr])
            y_train.append(usr2labels[usr])
        except: 
            print traceback.format_exc()
            raise
    return X_train, y_train
