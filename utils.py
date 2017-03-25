import sys
import traceback
import random
from config import DEBUG
from bisect import bisect_left
random.seed(87)

def debug(msg, val):
    if DEBUG > 0:
        print '[info] ' + msg, val

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
