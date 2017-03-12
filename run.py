import dataloaders.movielens100k as dataloader_movielens100k
import poolers.sample_pooler as sample_pooler
from utils import load_data, getDistribution, negativeSample, debug
from config import USR_TOTAL_LABELS_FIELDS, MAX_TRAIN_NUM, LAMBDA, LEARNING_RATE

import math
import numpy as np
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
    try:
        return 1 / (1 + math.exp(-x))
    except:
        # the only exception: x << 0 => exp(-x) explodes
        return 0.0

# here gradsOfW is a numpy 2D array
#      gradsOfV is {
#  someitemInd: itsGrad
#  ... }
def updateByGradients(W, V, gradsOfW, gradsOfV):
    W -= LEARNING_RATE * gradsOfW
    for itemInd in gradsOfV:
        V[itemInd, :] -= LEARNING_RATE * gradsOfV[itemInd]

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
                    bestCols = y_nonzeroCols_sample
    return bestCols

# get RMSE
def getRMSE(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler):
    totalSquaredErr = 0.0
    for usrid in usr2itemsIndx_valid:
        usr_rep = pooler.pool_all(usr2itemsIndx_valid[usrid], V)
        y_nonzeroCols = usr2NonzeroCols[usrid]

        # predict the most possible cols' combination
        bestCols = predictLabels(usr_rep, W)

        # calculate the error for such usrid
        squaredErr = ( len( list(set().union(bestCols, y_nonzeroCols)) ) - 3 ) * 2
        totalSquaredErr += squaredErr
    return math.sqrt(totalSquaredErr/len(usr2itemsIndx_valid))

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
   
def main():
    
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
    usr2itemsIndx, ind2itemNum = dataloader.load('data/usrItemRating_1st10k')
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
    usr2labels = dataloader.get_labels('data/usrAgeGenOccu')
    usr2NonzeroCols = dataloader.get_nonZeroCols('data/usrAgeGenOccu')
    print '[info] usr2labels, usr2NonzeroCols loaded'
    

    ''' acquire (k times) usr2NegativeSamples & usr2negsNonzeroCols ''' 
    cdfByLabels, labelsList = getDistribution(usr2labels)
    usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList, k=2)
    print '[info] usr2NegativeSamples, usr2negsNonzeroCols created'


    ''' init V to [-1, 1) ''' 
    numOfItems = len(ind2itemNum)
    itemFieldsNum = 4
    V = 2 * np.random.rand(numOfItems, itemFieldsNum) -1
    print '[info] V inited, V.shape == ', V.shape, '== (num items, itemFeatures length)'


    ''' init W to [-1, 1); init pooler'''
    # Warn: assume itemFieldsNum is the same after usr's representation's dimension
    totalLabelsNum = dataloader.gettotalLabelsNum()
    W = 2 * np.random.rand(itemFieldsNum, totalLabelsNum) -1
    pooler = sample_pooler.sample_pooler()
    print '[info] W & pooler inited, W.shape == ', W.shape, '== (itemFeatures length, total labels num)'
    debug('W', W)
    debug('V', V)


    ''' learning for W, V '''
    # Rmk, since no significant row over col (or the reverse); just 
    #   implement the most naive way
    # ref: http://stackoverflow.com/questions/17954990/performance-of-row-vs-column-operations-in-numpy
    rmse = 8787.0
    t = 0
    while rmse > 2 and t <= MAX_TRAIN_NUM:
        # TODO: if < some thresh, then negsample
        if False:
            cdfByLabels, labelsList = getDistribution(usr2labels)
            print 'cdfByLabels, labelsList', cdfByLabels, labelsList
            usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList, k=2)

        t += 1
        for usrid in usr2itemsIndx:
            # pooling
            usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
            debug('usr_rep', usr_rep)

            # get y, sumedW, negCoef
            # Warn: assume all usrs have all labels
            y = usr2labels[usrid]
            y_nonzeroCols = usr2NonzeroCols[usrid]
            sumedW = sumOverW(W, y_nonzeroCols)
            y_negSamples = usr2NegativeSamples[usrid]
            y_negsNonzeroCols = usr2negsNonzeroCols[usrid]
            grads = map(lambda sample: sigmoid(usr_rep.transpose().dot( sumOverW(W, sample) )), y_negsNonzeroCols)
            negCoef = reduce(lambda u, v: u+v, grads)

            # Warn: update W by usr, not by epoch 
            # get gradient of Wq (i.e. q-th column of W)
            gradsOfW = np.zeros(shape=W.shape)
            for q in range(W.shape[1]):
                # Warn: should be 2 * LAMBDA * ... (but just tune LAMBDA instead)
                gradVect2 = LAMBDA * W[:,[q]]
                WqGrads = gradVect2
                
                # if yq is 1 
                if q in y_nonzeroCols:
                    gradVect0 = (sigmoid(-1 * usr_rep.transpose().dot( sumedW )) * usr_rep).reshape(4,1)
                    gradVect1 = (negCoef * (-1 * usr_rep.transpose())).reshape(4,1)
                    WqGrads = WqGrads + gradVect0                     # <----- indep of which q
                    WqGrads = WqGrads + gradVect1                     # <----- indep of which q
                    debug('q', q)
                    debug('v0', gradVect0)
                    debug('v1', gradVect1)
                    debug('v2', gradVect2)
                    debug('wqgrads', WqGrads)

                gradsOfW[:,[q]] = WqGrads
            #debug('gradofw', gradsOfW)

            # Warn: update V by usr, not by epoch 
            # get gradient of Vitem, itemInd in X(usrid) 
            # Warn: only implemnet average poooling for now
            #   and the gradVect3, gradVect4 happen to hold the same values over all items in this case
            gradsOfV = {}
            itemLensInverse = 1.0 / len(usr2itemsIndx[usrid]) 
            for itemInd in usr2itemsIndx[usrid]:
                gradVect3 = sigmoid(-1 * usr_rep.transpose().dot(sumedW) ) * itemLensInverse * sumedW                                                                  # <--- indep of which item
                gradVect4 = negCoef * (-1 * itemLensInverse) * sumedW # <--- indep of which item
                # Warn: should be 2 * LAMBDA * ... (but just tune LAMBDA instead)
                gradVect5 = LAMBDA * V[itemInd,:]
                gradsOfV[itemInd] = gradVect3 + gradVect4 + gradVect5
                debug('v3', gradVect3)
                debug('v4', gradVect4)
                debug('v5', gradVect5)
                debug('gradsofv', gradsOfV[itemInd])
            debug('gradsofv', gradsOfV)

            # update gradients to W, V
            debug('Gw', gradsOfW)
            debug('Gv', gradsOfV)
            updateByGradients(W, V, gradsOfW, gradsOfV)
            #print '[info] after gradients updation: W[:,0] == ', W[:, 0], 'usr', usrid
            #print '[info] after gradients updation: V == ', V

        # valid
        rmse = getRMSE(W, V, usr2itemsIndx_valid, usr2NonzeroCols, pooler)
        print '[info]: rmse from validation == ', rmse

if __name__ == '__main__':
    main()
