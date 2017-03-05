import dataloaders.movielens100k as dataloader_movielens100k
import embedders.random_embedder as random_embedder
import poolers.sample_pooler as sample_pooler
from utils import load_data, getDistribution, negativeSample
from config import USR_TOTAL_LABELS_FIELDS, MAX_TRAIN_NUM, LAMBDA

import numpy as np
np.random.seed(123)  # Reproducibility

def checkConv(diff):
    return True #diff < 0.01

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

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# here gradsOfW is a numpy 2D array
#      gradsOfV is {
#  someitemInd: itsGrad
#  ... }
def updateByGradients(W, V, gradsOfW, gradsOfV):
    W += gradsOfW
    for itemInd in gradsOfV:
        V[itemInd, :] += gradsOfV[itemInd]
    print W, V
   
def main():
    
    ''' load each usr's BOI ''' 
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
    print '[info] usr2itemsIndx loaded'
    

    ''' init V ''' 
    numOfItems = len(ind2itemNum)
    itemFieldsNum = 4
    V = np.random.rand(numOfItems, itemFieldsNum)
    print '[info] V loaded, V.shape == ', V.shape, '== (num items, itemFeatures length)'


    ''' acquire usr2labels & usr2NonzeroCols ''' 
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
    

    ''' acquire usr2NegativeSamples & usr2negsNonzeroCols ''' 
    cdfByLabels, labelsList = getDistribution(usr2labels)
    usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList)
    print '[info] usr2NegativeSamples, usr2negsNonzeroCols created'


    ''' init W, pooler'''
    # Warn: assume itemFieldsNum is the same after usr's representation's dimension
    totalLabelsNum = dataloader.gettotalLabelsNum()
    W = np.random.rand(itemFieldsNum, totalLabelsNum)
    pooler = sample_pooler.sample_pooler()
    print '[info] W & pooler inited, W.shape == ', W.shape


    ''' learning for W, V '''
    # Rmk, since no significant row over col (or the reverse); just 
    #   implement the most naive way
    # ref: http://stackoverflow.com/questions/17954990/performance-of-row-vs-column-operations-in-numpy
    conv = False
    t = 0
    while not conv and t <= MAX_TRAIN_NUM:
        # TODO: if > thresh, then negsample
        if False:
            cdfByLabels, labelsList = getDistribution(usr2labels)
            print 'cdfByLabels, labelsList', cdfByLabels, labelsList
            usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList)

        t += 1
        for usrid in usr2itemsIndx:
            # pooling
            usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
            print 'usr_rep', usr_rep

            # get y, sumedW, negCoef
            # Warn: assume all usrs have all labels
            y = usr2labels[usrid]
            y_nonzeroCols = usr2NonzeroCols[usrid]
            sumedW = sumOverW(W, y_nonzeroCols)
            y_negSamples = usr2NegativeSamples[usrid]
            y_negsNonzeroCols = usr2negsNonzeroCols[usrid]
            #print 'sumedW', sumedW
            print usrid, y_negsNonzeroCols
            grads = map(lambda y_negNonzeroCols: sigmoid(usr_rep.transpose().dot( sumOverW(W, y_negNonzeroCols) )), y_negsNonzeroCols)
            negCoef = reduce(lambda u, v: u+v, grads)

            # Warn: update W, V by usr, not by epoch 
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
                    WqGrads += (gradVect0 + gradVect1)

                gradsOfW[:,[q]] = WqGrads

            # get gradient of Vitem, itemInd in X(usrid) 
            # Warn: only implemnet average poooling for now
            #   and the gradVect3, gradVect4 happen to hold the same values over all items in this case
            gradsOfV = {}
            itemLensInverse = 1.0 / len(usr2itemsIndx[usrid]) 
            for itemInd in usr2itemsIndx[usrid]:
                gradVect3 = sigmoid(-1 * usr_rep.transpose().dot(sumedW) ) * itemLensInverse * sumedW
                gradVect4 = negCoef * (-1 * itemLensInverse) * sumedW
                # Warn: should be 2 * LAMBDA * ... (but just tune LAMBDA instead)
                gradVect5 = LAMBDA * V[itemInd,:]
                gradsOfV[itemInd] = gradVect3 + gradVect4 + gradVect5

            # update gradients to W, V
            updateByGradients(W, V, gradsOfW, gradsOfV)

        # TODO: diff should route over trainData
        diff = 0.0
        conv = checkConv(diff)

if __name__ == '__main__':
    main()
