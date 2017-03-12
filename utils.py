import sys
import traceback
import random
from config import USR_LABELS_FIELDS, DEBUG
from bisect import bisect_left
random.seed(87)

def debug(msg, val):
    if DEBUG > 0:
        print '[info] ' + msg, val

# assign prepared the usr-labels file to each usr
# input file format (per line):
#    usrid|one-hot encoded feature_1|...|one-hot encoded feature_n
# TODO: modify this, only need list of nonzero col's ind 
#       hard-code the part for each col's num of rows
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
    usrids_train = []
    usrid2yInd = {} # take the indice of usr's y in y_train
    for usr in usr2representation:
        try:
            usrids_train.append[usr]
            X_train.append(usr2representation[usr])
            y_train.append(usr2labels[usr])
            usrid2yInd[usr] = len(y_train) - 1
        except: 
            print traceback.format_exc()
            raise
    return X_train, y_train, usrids_train

# return cdf (represented by cdfByLabels, labelsList)
# e.g. cdfByLabels = [0.3, 0.7]
# labelsList = [ [0,1], [1,0] ]
def getDistribution(usr2Labels):
    allUsrLabels = map(lambda usr: usr2Labels[usr], usr2Labels)
    labelsStr2Cnt = {}
    for labelVect in  allUsrLabels:
        labelsStr = ','.join([str(x) for x in labelVect])
        if labelsStr in labelsStr2Cnt:
            labelsStr2Cnt[labelsStr] += 1
        else:
            labelsStr2Cnt[labelsStr] = 1

    labelsList = [[int(x) for x in labelsStr.split(',')] for labelsStr in labelsStr2Cnt]
    cdfByLabels = []
    cnt = 0.0
    for labelsStr in labelsStr2Cnt:
        cnt += labelsStr2Cnt[labelsStr]
        cdfByLabels.append(cnt)
    cdfByLabels = map(lambda v: v/cnt, cdfByLabels)
    return cdfByLabels, labelsList

def keepNonzeroColInx(l):
    return [i for i, e in enumerate(l) if e != 0]

# do n times sampling, return thenm
def sample(cdfByLabels, labelsList, n):
    negativeLabels = []
    for i in range(n):
        # sample which 'labels'
        prob = random.random()
        ind = bisect_left(cdfByLabels, prob)
        negativeLabels.append(labelsList[ind])
    return negativeLabels

def negativeSample(usr2labels, cdfByLabels, labelsList, k=10):
    usr2NegativeSamples = {}
    usr2negsNonzeroCols = {}
    for usr in usr2labels:
        usr2NegativeSamples[usr] = sample(cdfByLabels, labelsList, n=k)
        usr2negsNonzeroCols[usr] = map(lambda x: keepNonzeroColInx(x), usr2NegativeSamples[usr])
    #print 'negativesamples, usr2negsNonzeroCols', usr2NegativeSamples, usr2negsNonzeroCols
    return usr2NegativeSamples, usr2negsNonzeroCols
