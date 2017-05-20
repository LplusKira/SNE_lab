import dataloaders.movielens100k as dataloader_movielens100k
import poolers.sample_pooler as sample_pooler
from utils import getDistribution, negativeSample, debug
from config import USR_TOTAL_LABELS_FIELDS, ITEM_FIELDS_NUM, MAX_TRAIN_NUM, LAMBDA, LEARNING_RATE, MOMENTUM, NEG_SAMPLE_NUM

import math, random, sys, traceback
import numpy as np
from time import gmtime, strftime

np.random.seed(0)  # Reproducibility

# TODO: modify this: hard code num of fields + each field's size(category num)
# ref: http://stackoverflow.com/questions/8386675/extracting-specific-columns-in-numpy-array
def takeNonZero(W, y):
    # y = list of nonzeor col's indice
    return W[:,y]

# return 1-dim by summing 'nonzeor cols'(the corresponding one-hot val is nonzero) of W
def sumOverW(W, y):
    # Wn notes the 'nonzero cols' of W's concatenated matrix
    Wn = takeNonZero(W, y)
    return Wn.sum(axis=1)   

def sigmoid(x):
    #return 1 / (1 + math.exp(-x))
    try:
        return 1 / (1 + math.exp(-x))
    except:
        # the only exception: x << 0 => exp(-x) explodes
        return 0.001
        #print traceback.format_exc()
        #sys.exit(1)

# here gradsOfW is a numpy 2D array
#      gradsOfV is {
#          someitemInd: itsGrad
#      ... }
def updateByGradients(W, V, gradsOfW, gradsOfV, incrInd):
    scale = MOMENTUM if incrInd else 1/MOMENTUM
    W -= scale * LEARNING_RATE * gradsOfW
    for itemInd in gradsOfV:
        V[itemInd, :] -= scale * LEARNING_RATE * gradsOfV[itemInd]
    return W, V

def predictLabels(usr_rep, W):
    # XXX, its hard-coded and inefficient now
    bestCols = [0, 4, 6]
    bestScore = usr_rep.transpose().dot( sumOverW(W, bestCols) )
    for i1 in range(4):
        for i2 in range(4, 4+2):
            for i3 in range(6, 6+21):
                y_nonzeroCols_sample = [i1, i2, i3]
                sumedW = sumOverW(W, y_nonzeroCols_sample)
                score = usr_rep.transpose().dot( sumedW )
                if score >= bestScore:
                    bestScore = score
                    bestCols = y_nonzeroCols_sample
    return bestCols

# get RMSE
def getRMSE(W, V, usr2itemsIndx, usr2NonzeroCols, pooler):
    totalSquaredErr = 0.0
    for usrid in usr2itemsIndx:
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
        y_nonzeroCols = usr2NonzeroCols[usrid]

        # predict the most possible cols' combination
        bestCols = predictLabels(usr_rep, W)

        # calculate the error for such usrid
        squaredErr = ( len( list(set().union(bestCols, y_nonzeroCols)) ) - 3 ) * 2
        totalSquaredErr += squaredErr
    return math.sqrt(totalSquaredErr/len(usr2itemsIndx))

# return {
#  0: tp list(of cols) 
#  1: fp list(of cols)
#  2: tn list(of cols)
# }
def getClasses(trueCols, predictedCols):
    # XXX hard coded
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

# ref(how to cal microf1): http://rushdishams.blogspot.tw/2011/08/micro-and-macro-average-of-precision.html
# get micro f1 by age/gender/occupation
def getMicroF1ByCol(W, V, usr2itemsIndx, usr2NonzeroCols, pooler):
    # XXX hard coded: age/gender/occupation
    tpList = [0.0] * USR_TOTAL_LABELS_FIELDS
    fpList = [0.0] * USR_TOTAL_LABELS_FIELDS
    tnList = [0.0] * USR_TOTAL_LABELS_FIELDS
    for usrid in usr2itemsIndx:
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
        y_nonzeroCols = usr2NonzeroCols[usrid]

        # predict the most possible cols' combination
        bestCols = predictLabels(usr_rep, W)

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
    microPrecision = summedTp/ (summedTp + sum(fpList))
    microRecall    = summedTp/ (summedTp + sum(tnList))

    # cal micro F1
    microF1        = 2 * microPrecision * microRecall / (microPrecision + microRecall) if(summedTp > 0) else 0.0
    return microF1

# get one error
#   one error = sum( has one class hits or not ) / dataPointsNum
def getOneError(W, V, usr2itemsIndx, usr2NonzeroCols, pooler):
    errCnt = len(usr2itemsIndx)
    usrCnt = len(usr2itemsIndx)
    for usrid in usr2itemsIndx:
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
        y_nonzeroCols = usr2NonzeroCols[usrid]

        # predict the most possible cols' combination
        bestCols = predictLabels(usr_rep, W)
        for ind, col in enumerate(bestCols):
            if col == y_nonzeroCols[ind]:
                # if one class(col) hits, then no err for this usr
                errCnt -= 1
                break
    return errCnt / float(usrCnt)
                
# get RL (ranking loss)
#   it's .. ??? 
# TODO: may modify the way this calculates .. inefficient now
def getRL(W, V, usr2itemsIndx, usr2NonzeroCols, pooler):
    totalLoss = 0.0
    for usrid in usr2itemsIndx:
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
        y_nonzeroCols = usr2NonzeroCols[usrid]
        y_zeroCols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        col2Val = map(lambda v: [v, 0], y_zeroCols)
        for v in y_nonzeroCols:
            y_zeroCols.remove(v)

        # predict the most possible cols' combination + add to sort list
        bestCols = predictLabels(usr_rep, W)
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
        lossPerUsr = errCnt / 72.0 # <-- hard-coded here
        totalLoss += lossPerUsr

    return totalLoss / len(usr2itemsIndx)
        
# get coverage 
#   covreage = find the last one's position (ranked by predicted probability)
#   we may assume 0 1 | 1 0 0:   prob pos(1) >= prob pos(0); prob pos(2) >= prob pos(3), prob pos(2) >= prob pos(4),
#             pos 0 1   2 3 4 
#   but have no other knowledge, so 'sort by prob' would just have: 1 1 0 0 0 (i.e. doesnt change its original ordery)
#                                                                   1 2 0 3 4
def getCoverage(W, V, usr2itemsIndx, usr2NonzeroCols, pooler):
    totalFields = 1.0 * USR_TOTAL_LABELS_FIELDS
    colNums = len( next(usr2NonzeroCols.itervalues()) )
    loss = 0.0
    for usrid in usr2itemsIndx:
        y_nonzeroCols = usr2NonzeroCols[usrid]

        # predict the most possible cols' combination
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
        bestCols = predictLabels(usr_rep, W)
           
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

    return loss / len(usr2itemsIndx)


# get average precision 
#   we may assume 0 1 | 1 0 0:   prob pos(1) >= prob pos(0); prob pos(2) >= prob pos(3), prob pos(2) >= prob pos(4),
#             pos 0 1   2 3 4
#   since we still dont have each field's prob
#     so 'sort by prob' would just have: 1 1 0 0 0 (i.e. doesnt change its original ordery)
#                                        1 2 0 3 4
def getAvgPrecision(W, V, usr2itemsIndx, usr2NonzeroCols, pooler):
    colNums = len( next(usr2NonzeroCols.itervalues()) )
    prec = 0.0
    for usrid in usr2itemsIndx:
        y_nonzeroCols = usr2NonzeroCols[usrid]

        # predict the most possible cols' combination
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
        bestCols = predictLabels(usr_rep, W)
           
        # each 'one' has a value: (its reverse pos + 1 in 'ones' by prob) / (its reverse pos + 1 in all fields by prob)
        #                              ^^ i.e. higher porb has lowe pos
        col2AllRank = {}
        score = 0.0 
        for cnt in range(0, colNums):
            ind = colNums - 1 - cnt
            if y_nonzeroCols[ind] == bestCols[ind]:
              col2AllRank[ y_nonzeroCols[ind] ] = ind + 1
            else:
              col = y_nonzeroCols[ind]
              col2AllRank[ col ] = col + len(filter(lambda v: v > col, bestCols))  + 1

        # sort by Allrank lower to bigger 
        rankedList = sorted(col2AllRank.items(), key=lambda x: x[1])
        for ind, val in enumerate(rankedList):
          score += float(ind + 1) / val[1]

        prec += score / colNums
                
    return prec / len(usr2itemsIndx)

# get  hamming loss
#   we may assume 0 1 | 1 0 0:  pred 
#                 1 0 | 0 1 0:  real 
#   since the papaer itself doesnt specify, we use pred XOR(by attribute) real (i.e. 1 | 1 => hamming loss (for this): 2/2)
def getHammingLoss(W, V, usr2itemsIndx, usr2NonzeroCols, pooler):
    colNums = len( next(usr2NonzeroCols.itervalues()) )
    loss = 0.0
    for usrid in usr2itemsIndx:
        y_nonzeroCols = usr2NonzeroCols[usrid]

        # predict the most possible cols' combination
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
        bestCols = predictLabels(usr_rep, W)
           
        dataPointLoss = 0.0
        for ind, val in enumerate(bestCols):
            if bestCols[ind] != y_nonzeroCols[ind]:
                dataPointLoss += 1.0
    
        loss += dataPointLoss / colNums
    return loss / len(usr2itemsIndx)



# just print the result
def printTruePredicted(W, V, usr2itemsIndx, usr2NonzeroCols, pooler):
    print '[info]: usrid, actual, predicted'
    for usrid in usr2itemsIndx:
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
        y_nonzeroCols = usr2NonzeroCols[usrid]

        # predict the most possible cols' combination
        bestCols = predictLabels(usr_rep, W)

        print usrid, y_nonzeroCols, bestCols


# pick the 1st 10% as valid data
def splitTrainTest(usr2itemsIndx):
    totalUsrs = len(usr2itemsIndx)
    cnt = 0
    usr2itemsIndx_train = {}
    usr2itemsIndx_valid = {}
    for usr in usr2itemsIndx:
        if cnt > totalUsrs * 0.1:
            usr2itemsIndx_train[usr] = usr2itemsIndx[usr]
        else:
            usr2itemsIndx_valid[usr] = usr2itemsIndx[usr]
        cnt += 1

    return usr2itemsIndx_train, usr2itemsIndx_valid

# formulate avg loss
def getAvgLoss(W, V, usr2NonzeroCols, usr2negsNonzeroCols, usr2itemsIndx, pooler):
    loss = 0.0
    cnt = 0
    for usrid in usr2itemsIndx:
        try: 
            usrloss = 0.0 
            usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)

            # 0. -log( sigmoid( usr_rep * sumedW_y) )
            y_nonzeroCols = usr2NonzeroCols[usrid]
            sumedW_y = sumOverW(W, y_nonzeroCols)
            usrloss += (-1) * math.log( sigmoid(usr_rep.transpose().dot(sumedW_y)) )

            # 1. summation log( sigmoid( usr_rep * sumedW_neg ) )
            y_negsNonzeroCols = usr2negsNonzeroCols[usrid]
            sumedW_negs = map(lambda y_negNonzeroCols: sumOverW(W, y_negNonzeroCols).reshape(ITEM_FIELDS_NUM,1), y_negsNonzeroCols)
            usrloss += (-1) * sum(map(lambda sumedW_neg: math.log( sigmoid( (-1) * usr_rep.transpose().dot(sumedW_neg) ) ), sumedW_negs))

            # 2. l2 norm
            l2norm = np.linalg.norm(W) + sum(map(lambda v: np.linalg.norm(v), V))
            usrloss += l2norm

            loss += usrloss
            cnt += 1
        except:
            loss += 0.0
            cnt += 0
    return loss/cnt

# get sampled (by usr) loss (cause getAvgLoss is pretty slow)
def getSampledLoss(W, V, usr2NonzeroCols, usr2negsNonzeroCols, usr2itemsIndx, pooler, sampleSize = 10):
    loss = 0.0
    usrs = list(usr2itemsIndx.keys())
    sampledUsrsIndice = random.sample(range(len(usrs)), sampleSize)  # sample without replacement
    sampledUsrs = map(lambda ind: usrs[ind], sampledUsrsIndice)
    for usrid in sampledUsrs:
        usrloss = 0.0 
        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)

        # 0. -log( sigmoid( usr_rep * sumedW_y) )
        y_nonzeroCols = usr2NonzeroCols[usrid]
        sumedW_y = sumOverW(W, y_nonzeroCols)
        usrloss += (-1) * math.log( sigmoid(usr_rep.transpose().dot(sumedW_y)) )

        # 1. summation log( sigmoid( usr_rep * sumedW_neg ) )
        y_negsNonzeroCols = usr2negsNonzeroCols[usrid]
        sumedW_negs = map(lambda y_negNonzeroCols: sumOverW(W, y_negNonzeroCols).reshape(ITEM_FIELDS_NUM,1), y_negsNonzeroCols)
        usrloss += (-1) * sum(map(lambda sumedW_neg: math.log( sigmoid( (-1) * usr_rep.transpose().dot(sumedW_neg) ) ), sumedW_negs))

        # 2. l2 norm
        l2norm = np.linalg.norm(W) + sum(map(lambda v: np.linalg.norm(v), V))
        usrloss += l2norm

        loss += usrloss
    return loss

# return gradsOfW w.r.t. the user
''' y_nonzeroCols:    usr's y's nonzero cols ''' 
''' sigmoid_y:        sigmoid(-v * summedW_y ) ''' 
''' usr_rep:          usr's representation vector ''' 
''' sigmoids_negs:    [ sigmoid( v * summedW_neg ) ] '''
''' y_negsNonzeroCols:[ nonzeroCols ] ''' 
def getGradsOfW(W, y_nonzeroCols, sigmoid_y, usr_rep, sigmoids_negs, y_negsNonzeroCols):
    gradsOfW = np.zeros(shape=W.shape)

    # get each Wq's gradient
    for q in range(W.shape[1]):
        gradVect2 = 2 * LAMBDA * W[:,[q]]
        WqGrads   = gradVect2
        
        # if yq is 1 
        gradVect0 = np.zeros( (ITEM_FIELDS_NUM,1) )
        if q in y_nonzeroCols:
            gradVect0 = (-1) * (sigmoid_y * usr_rep).reshape(ITEM_FIELDS_NUM,1)
            WqGrads += gradVect0                     # <----- indep of which q ( if in y's cols )

        # iterate over negs
        negCoef = 0.0
        for ind, y_negNonzeroCols in enumerate(y_negsNonzeroCols):
            if q in y_negNonzeroCols:
                negCoef += sigmoids_negs[ind]
        gradVect1 = (negCoef * usr_rep.transpose()).reshape(ITEM_FIELDS_NUM,1)
        WqGrads += gradVect1                         # <----- indep of which q ( if in y_neg's cols )

        gradsOfW[:,[q]] = WqGrads
    return gradsOfW

# return gradsOfV w.r.t. the user
def getGradsOfV(V, itemsIndx, sumedW_y, sigmoid_y, sigmoidedSumedW):
    gradsOfV = {}
    itemLens = len(itemsIndx) 
    for itemInd in itemsIndx:
        gradVect3 = (-1) * (1.0 / itemLens) * sigmoid_y * sumedW_y                   # <--- indep of which item
        gradVect4 = ((1.0 / itemLens) * sigmoidedSumedW).transpose()                 # <--- indep of which item
        gradVect5 = 2 * LAMBDA * V[itemInd,:]
        gradsOfV[itemInd] = (gradVect3 + gradVect4 + gradVect5).reshape(ITEM_FIELDS_NUM,)         # <-- reshape to vect (not 2D array)

    return gradsOfV

def getTerms(usrid, usr2labels, usr2NonzeroCols, usr2itemsIndx, W, usr_rep, usr2negsNonzeroCols):
    y                 = usr2labels[usrid]
    y_nonzeroCols     = usr2NonzeroCols[usrid]
    itemsIndx         = usr2itemsIndx[usrid]
    sumedW_y          = sumOverW(W, y_nonzeroCols)
    sigmoid_y         = sigmoid((-1) * usr_rep.transpose().dot(sumedW_y)) # equivalent to 1 - sigmoid(...)
    y_negsNonzeroCols = usr2negsNonzeroCols[usrid]
    sumedW_negs       = map(lambda y_negNonzeroCols: sumOverW(W, y_negNonzeroCols).reshape(ITEM_FIELDS_NUM,1), y_negsNonzeroCols)
    sigmoids_negs     = map(lambda sumedW_neg: sigmoid(usr_rep.transpose().dot(sumedW_neg)), sumedW_negs)
    sigmoidedSumedW   = np.zeros((ITEM_FIELDS_NUM,1))
    for ind, sigmoid_neg in enumerate(sigmoids_negs):
        sigmoidedSumedW += sigmoid_neg * sumedW_negs[ind]
    return y, y_nonzeroCols, itemsIndx, sumedW_y, sigmoid_y, y_negsNonzeroCols, sumedW_negs, sigmoids_negs, sigmoidedSumedW
   
def main(argv):
    if not len(argv) == 1:
        print '[info] usage: python run.py yourtraindata'
        return 1
    
    ''' load each usr's BOI (and for valid data) ''' 
    # sample: 
    #  ind2itemNum = {
    #    0: 5566,
    #    1: 87,
    #    2: 2266,
    #  }
    #  usr2itemsIndx = {
    #    0: [0,1],
    #    1: [1,2],
    #  }
    # Rmk: the part for 'filtering rating >= 3' is done in file
    dataloader = dataloader_movielens100k.dataloader_movielens100k()
    usr2itemsIndx, ind2itemNum = dataloader.load(argv[0])
    usrs = map(lambda usr: usr, usr2itemsIndx)
    usr2itemsIndx, usr2itemsIndx_valid = splitTrainTest(usr2itemsIndx)
    print '[info] usr2itemsIndx, usr2itemsIndx_valid loaded'
    print '[info] usrs in train: ', len(usr2itemsIndx)
    print '[info] usrs in valid: ', len(usr2itemsIndx_valid)
    
    
    ''' acquire (for all usrs) usr2labels & usr2NonzeroCols ''' 
    # sample:
    # usr2labels = {
    #   0: [0,0,1, 1,0],
    #   1: [1,0,0, 0,1],
    # }
    # usr2NonzeroCols = {
    #   0: [2, 3],
    #   1: [0, 4],
    # }
    usr2labels = dataloader.get_labels('data/usrAgeGenOccu', usrs)
    usr2NonzeroCols = dataloader.get_nonZeroCols('data/usrAgeGenOccu')
    print '[info] usr2labels, usr2NonzeroCols loaded'
    

    ''' acquire (k times) usr2NegativeSamples & usr2negsNonzeroCols ''' 
    cdfByLabels, labelsList = getDistribution(usr2labels)
    usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList, k=NEG_SAMPLE_NUM)
    print '[info] usr2NegativeSamples, usr2negsNonzeroCols created'


    ''' init V to [-1, 1) ''' 
    numOfItems = len(ind2itemNum)
    V = 2 * np.random.rand(numOfItems, ITEM_FIELDS_NUM) -1
    print '[info] V inited, V.shape == ', V.shape, '== (num items, itemFeatures length)'


    ''' init W to [-1, 1); init pooler'''
    # Warn: assume ITEM_FIELDS_NUM is the same after usr's representation's dimension
    totalLabelsNum = dataloader.gettotalLabelsNum()
    W = 2 * np.random.rand(ITEM_FIELDS_NUM, totalLabelsNum) -1
    pooler = sample_pooler.sample_pooler()
    print '[info] W & pooler inited, W.shape == ', W.shape, '== (itemFeatures length, total labels num)'
    debug('W', W)
    debug('V', V)


    ''' learning for W, V '''
    # Rmk, since no significant row over col (or the reverse); just 
    #   implement the most naive way
    # ref: http://stackoverflow.com/questions/17954990/performance-of-row-vs-column-operations-in-numpy
    microF1Valid = 0.0 # <--- btw [0,1]; bigger => higer precision, higer recall
    prevAvgLoss  = float('inf')
    lossDiff     = float('inf')
    t            = 0
    incrInd      = False
    #while t <= MAX_TRAIN_NUM:
    while microF1Valid < 0.5 and t <= MAX_TRAIN_NUM:
        t += 1
        print '\n[info] ######################## '
        print '[info] run == ', t
        print '[info] ######################## '
        print '[info] time == ', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # if > some run OR |loss' diff| small, then negsample
        if (t >= 100 and t == MAX_TRAIN_NUM * 0.9) or ( math.fabs(lossDiff) < 1e-6 and t % 5 == 0 ):
            print '[info] resample usr negative samples'
            usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList, k=NEG_SAMPLE_NUM)
            print '[info] diff', lossDiff
            print '[info] usr2NegativeSamples: '
            for usrid in usr2NegativeSamples:
                print usrid
                for sample in usr2NegativeSamples[usrid]:
                    print ' ', sample

        for usrid in usr2itemsIndx:
            # pooling
            usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)

            # get y, sumedW(for y AND negs), sigmoids(for y AND negs)
            # Warn: assume all usrs have all labels
            y, y_nonzeroCols, itemsIndx, sumedW_y, sigmoid_y, y_negsNonzeroCols, sumedW_negs, sigmoids_negs, sigmoidedSumedW = getTerms(usrid, usr2labels, usr2NonzeroCols, usr2itemsIndx, W, usr_rep, usr2negsNonzeroCols)

            # Warn: update W by usr, not by epoch 
            # get gradient of Wq (i.e. q-th column of W)
            gradsOfW = getGradsOfW(W, y_nonzeroCols, sigmoid_y, usr_rep, sigmoids_negs, y_negsNonzeroCols)
            
            # Warn: update V by usr, not by epoch 
            # get gradient of Vitem, itemInd in X(usrid) 
            # Warn: only implemnet average poooling for now
            #   and the gradVect3, gradVect4 happen to hold the same values over all items in this case
            gradsOfV = getGradsOfV(V, itemsIndx, sumedW_y, sigmoid_y, sigmoidedSumedW)

            # update gradients to W, V
            W, V = updateByGradients(W, V, gradsOfW, gradsOfV, incrInd)

            if usrid % 20 == 0:
                print '[info] usr', usrid, 'l2 norm of gradsOfW == ', np.linalg.norm(gradsOfW)

        if t % 5 == 0:
            avgloss = getAvgLoss(W, V, usr2NonzeroCols, usr2negsNonzeroCols, usr2itemsIndx, pooler)
            lossDiff = avgloss - prevAvgLoss 
            prevAvgLoss = avgloss
            print '[info] at run == ', t
            print '[info] avgloss == ', avgloss, '(only for train -- since this requires samples)'
            print '[info] delta loss == ', lossDiff
            print '[info] W[:,0] == ', W[:, 0]
            print '[info] V == ', V

        # check performance by microF1, oneError
        microF1Train = getMicroF1ByCol(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        microF1Valid = getMicroF1ByCol(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler)
        oneErrorTrain = getOneError(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        oneErrorValid = getOneError(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler)
        RLTrain = getRL(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        RLValid = getRL(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler)
        coverageTrain = getCoverage(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        coverageValid = getCoverage(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler)
        avgPrecTrain = getAvgPrecision(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        avgPrecValid = getAvgPrecision(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler)
        HLTrain = getHammingLoss(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        HLValid = getHammingLoss(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler)
        
        print '[info] train data microF1 == ', microF1Train
        print '[info] valid data microF1 == ', microF1Valid
        print '[info] train data oneError == ', oneErrorTrain
        print '[info] valid data oneError == ', oneErrorValid
        print '[info] train data RL == ', RLTrain
        print '[info] valid data RL == ', RLValid
        print '[info] train data coverage == ', coverageTrain
        print '[info] valid data coverage == ', coverageValid
        print '[info] train data avgPrec == ', avgPrecTrain
        print '[info] valid data avgPrec == ', avgPrecValid
        print '[info] train data hammingLoss == ', HLTrain
        print '[info] valid data hammingLoss == ', HLValid

    print '[info]: for traindata, print real vals & predicted vals ... '
    printTruePredicted(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
    print '[info]: for validdata, print real vals & predicted vals ... '
    printTruePredicted(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler)

if __name__ == '__main__':
    main(sys.argv[1:])
