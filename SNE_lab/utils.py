import random
import math
from bisect import bisect_left
random.seed(87)


# Return cdf (represented by cdfByLabels, labelsList)
# e.g. cdfByLabels = [0.3, 0.7 + 0.3]
# labelsList = [ [0,1], [1,0] ]
def getDistribution(usr2Labels):
    allUsrLabels = map(lambda usr: usr2Labels[usr], usr2Labels)
    labelsStr2Cnt = {}
    for labelVect in allUsrLabels:
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


# Do n times sampling (must unequal to usrLabels), return them
def sample(cdfByLabels, labelsList, usrLabels, n):
    negativeLabels = []
    for i in range(n):
        # sample which 'labels'
        sampledLabels = usrLabels
        while sampledLabels == usrLabels:
            prob = random.random()
            ind = bisect_left(cdfByLabels, prob)
            sampledLabels = labelsList[ind]
        negativeLabels.append(labelsList[ind])
    return negativeLabels


def negativeSample(usr2labels, cdfByLabels, labelsList, k=10):
    def keepNonzeroColInx(l):
        return [i for i, e in enumerate(l) if e != 0]

    usr2NegativeSamples = {}
    usr2negsNonzeroCols = {}
    for usr in usr2labels:
        usrLabels = usr2labels[usr]
        usr2NegativeSamples[usr] = sample(cdfByLabels, labelsList, usrLabels, n=k)
        usr2negsNonzeroCols[usr] = map(lambda x: keepNonzeroColInx(x), usr2NegativeSamples[usr])
    return usr2NegativeSamples, usr2negsNonzeroCols


# Split a dict to k disjoint subdicts (by keys)
def splitKfolds(d, k):
    Keys = d.keys()
    Kfolds = [{} for v in range(k)]
    random.shuffle(Keys)

    for ind, key in enumerate(Keys):
        Kfolds[ind % k][key] = d[key]
    return Kfolds


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# Warn:
#   This 'sigmoid' is 'pseudo sigmoid'
#   It returns 0.001 when x is extremely small
#   It was meant to handle exposion such as log(sigmoid(x)); but, it's a bad design
#   Implement your own if needed!
# Return 1 / (1 + math.exp(-x)) || 0.001
def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        # Calculation exception: x << 0 => exp(-x) explodes
        return 0.001


# Return 1-dim ndarray by summing 'nonzeor cols' (by y) of W
def sumOverW(W, y):
    # Ref: https://stackoverflow.com/a/8386737
    # Wn: 'nonzero cols' of W's concatenated matrix
    Wn = W[:, y]
    return Wn.sum(axis=1)


def getMicroF1ByCol(args):
    '''Get microF1
    ref: http://rushdishams.blogspot.tw/2011/08/micro-and-macro-average-of-precision.html
    '''
    usr2NonzeroCols = args['usr2NonzeroCols']
    u2predictions = args['u2predictions']
    totalLabelsNum = args['totalLabelsNum']

    def getClasses(trueCols, predictedCols):
        # return {
        #  0: tp list(of cols)
        #  1: fp list(of cols)
        #  2: tn list(of cols)
        # }
        classDict = {
          0: [],
          1: [],
          2: [],
        }
        for col in trueCols:
            if col in predictedCols:
                classDict[0].append(col)
            else:
                classDict[2].append(col)
        for col in predictedCols:
            if col not in trueCols:
                classDict[1].append(col)
        return classDict

    tpList = [0.0] * totalLabelsNum
    fpList = [0.0] * totalLabelsNum
    tnList = [0.0] * totalLabelsNum
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]

        # update tp, fp, tn
        # 0: tp, 1: fp, 2: tn
        classDict = getClasses(y_nonzeroCols, bestCols)
        for col in classDict[0]:
            tpList[col] += 1.0
        for col in classDict[1]:
            fpList[col] += 1.0
        for col in classDict[2]:
            tnList[col] += 1.0

    # cal micro precision & recall
    #   micor precision = sum(tp) / (sum(tp) + sum(fp))
    #   micro recall    = sum(tp) / (sum(tp) + sum(tn))
    summedTp = sum(tpList)
    microPrecision = summedTp / (summedTp + sum(fpList))
    microRecall = summedTp / (summedTp + sum(tnList))

    # cal micro F1
    microF1 = 2 * microPrecision * microRecall / (microPrecision + microRecall) if(summedTp > 0) else 0.0
    return microF1


def getOneError(args):
    '''Get one error
    sum( has one class hits or not ) / dataPointsNum
    '''
    usr2NonzeroCols = args['usr2NonzeroCols']
    u2predictions = args['u2predictions']

    errCnt = len(u2predictions)
    usrCnt = len(u2predictions)
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]
        for ind, col in enumerate(bestCols):
            if col == y_nonzeroCols[ind]:
                # if one class(col) hits, then no err for this usr
                errCnt -= 1
                break
    return errCnt / float(usrCnt)


def getRL(args):
    '''Get RL (ranking loss)
    examine each (0,1) pair's order in the sorted cols in prediction (by prob)

    Notice for 'sort by probability's adaptation in SNE:
        Example 0 1 | 1 0 0:
           (pos 0 1   2 3 4)
            we can assert at most:
            prob pos(1) >= prob pos(0)
            prob pos(2) >= prob pos(3)
            prob pos(2) >= prob pos(4)
        Since no further knowledge, so 'sort by prob' renders:
            (by stable sort)
            1 1 0 0 0
            1 2 0 3 4 (pos)
    '''
    usr2NonzeroCols = args['usr2NonzeroCols']
    u2predictions = args['u2predictions']
    totalLabelsNum = args['totalLabelsNum']
    rlPairsCnt = args['rlPairsCnt']

    totalLoss = 0.0
    rlPairsCnt = float(rlPairsCnt)
    for usrid in u2predictions:
        bestCols = u2predictions[usrid]
        y_nonzeroCols = usr2NonzeroCols[usrid]

        y_zeroCols = range(totalLabelsNum)
        col2Val = map(lambda v: [v, 0], y_zeroCols)
        for v in y_nonzeroCols:
            y_zeroCols.remove(v)

        for v in bestCols:
            col2Val[v][1] = 1
        col2Val.sort(key=lambda v: v[1], reverse=True)
        col2Order = {}
        for ind, v in enumerate(col2Val):
            col2Order[v[0]] = ind

        # check for every true's '0','1''s indx pair
        # if the ordery of predicted is reverse => err ++
        errCnt = 0
        for col1 in y_zeroCols:
            for col2 in y_nonzeroCols:
                if col2Order[col1] < col2Order[col2]:
                    errCnt += 1
        lossPerUsr = errCnt / rlPairsCnt
        totalLoss += lossPerUsr

    return totalLoss / len(u2predictions)


def getCoverage(args):
    '''Get coverage
    find the last one's position (ranked by predicted probability)

    Similar notice here as in getRL
    '''
    usr2NonzeroCols = args['usr2NonzeroCols']
    u2predictions = args['u2predictions']
    totalLabelsNum = args['totalLabelsNum']

    totalFields = 1.0 * totalLabelsNum
    colNums = len(next(usr2NonzeroCols.itervalues()))
    loss = 0.0
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]

        # rank by prob (start from 0): i.e. lowest prob => bigger rank number
        lowestOneRank = colNums - 1
        for cnt in range(0, colNums):
            ind = colNums - 1 - cnt
            if bestCols[ind] > y_nonzeroCols[ind]:
                lowestOneRank = y_nonzeroCols[ind] + cnt + 1
                break
            elif bestCols[ind] < y_nonzeroCols[ind]:
                lowestOneRank = y_nonzeroCols[ind] + cnt
                break

        loss += lowestOneRank / totalFields

    return loss / len(u2predictions)


def getAvgPrecision(args):
    '''Get average precision

    Similar notice here as in getRL
    '''
    usr2NonzeroCols = args['usr2NonzeroCols']
    u2predictions = args['u2predictions']

    colNums = len(next(usr2NonzeroCols.itervalues()))
    prec = 0.0
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]

        # each 'real one' has a value: (its reverse pos + 1 in 'ones' by prob) / (its reverse pos + 1 in all fields by prob)
        #                              ^^ i.e. higher porb has lowe pos
        col2AllRank = {}
        score = 0.0
        for cnt in range(0, colNums):
            ind = colNums - 1 - cnt
            if y_nonzeroCols[ind] == bestCols[ind]:
                col2AllRank[y_nonzeroCols[ind]] = ind + 1
            else:
                col = y_nonzeroCols[ind]
                col2AllRank[col] = col + len(filter(lambda v: v > col, bestCols)) + 1

        # sort by Allrank lower to bigger
        rankedList = sorted(col2AllRank.items(), key=lambda x: x[1])
        for ind, val in enumerate(rankedList):
            score += float(ind + 1) / val[1]

        prec += score / colNums

    return prec / len(u2predictions)


def getHammingLoss(args):
    '''Get hamming loss
    explain by an example

    Example 0 1 | 1 0 0:  pred
            1 0 | 0 1 0:  real
    We use pred XOR (by attribute) real
    Thus, hamming loss for this case is 2/2
    '''
    usr2NonzeroCols = args['usr2NonzeroCols']
    u2predictions = args['u2predictions']

    colNums = len(next(usr2NonzeroCols.itervalues()))
    loss = 0.0
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]

        dataPointLoss = 0.0
        for ind, val in enumerate(bestCols):
            if bestCols[ind] != y_nonzeroCols[ind]:
                dataPointLoss += 1.0

        loss += dataPointLoss / colNums
    return loss / len(u2predictions)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
