from utils import sigmoid, sumOverW
from itertools import product
import numpy as np
from sys import path
path.append('../')  # To import utils funcs
import math, random

# Update/get loss, model params, current predictions statelessly
class Baseupdator(object):
    def __init__(self, *args):
        self.NEG_SAMPLE_NUM = args[0]
        self.ITEM_FIELDS_NUM = args[1]
        self.MAX_TRAIN_NUM = args[2]
        self.LEARNING_RATE = args[3]
        self.MOMENTUM = args[4]
        self.LAMBDA = args[5]

    # XXX employ ndarray op???
    # Formulate avg loss
    def getAvgLoss(self, W, V, usr2NonzeroCols, usr2negsNonzeroCols, usr2itemsIndx, pooler):
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
                sumedW_negs = map(lambda y_negNonzeroCols: sumOverW(W, y_negNonzeroCols).reshape(self.ITEM_FIELDS_NUM,1), y_negsNonzeroCols)
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

    # XXX employ ndarray op???
    # Get sampled (by usr) loss 
    # - Cauz getAvgLoss is SLOW
    def getSampledLoss(self, W, V, usr2NonzeroCols, usr2negsNonzeroCols, usr2itemsIndx, pooler, sampleSize = 10):
    
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
            sumedW_negs = map(lambda y_negNonzeroCols: sumOverW(W, y_negNonzeroCols).reshape(self.ITEM_FIELDS_NUM,1), y_negsNonzeroCols)
            usrloss += (-1) * sum(map(lambda sumedW_neg: math.log( sigmoid( (-1) * usr_rep.transpose().dot(sumedW_neg) ) ), sumedW_negs))
    
            # 2. l2 norm
            l2norm = np.linalg.norm(W) + sum(map(lambda v: np.linalg.norm(v), V))
            usrloss += l2norm
    
            loss += usrloss
        return loss

    # Return gradsOfW w.r.t. the user
    ''' y_nonzeroCols:    usr's y's nonzero cols ''' 
    ''' sigmoid_y:        sigmoid(-v * summedW_y ) ''' 
    ''' usr_rep:          usr's representation vector ''' 
    ''' sigmoids_negs:    [ sigmoid( v * summedW_neg ) ] '''
    ''' y_negsNonzeroCols:[ nonzeroCols ] ''' 
    def getGradsOfW(self, W, y_nonzeroCols, sigmoid_y, usr_rep, sigmoids_negs, y_negsNonzeroCols):
    
        gradsOfW = np.zeros(shape=W.shape)
    
        # get each Wq's gradient
        for q in range(W.shape[1]):
            gradVect2 = 2 * self.LAMBDA * W[:,[q]]
            WqGrads   = gradVect2
            
            # if yq is 1 
            gradVect0 = np.zeros( (self.ITEM_FIELDS_NUM,1) )
            if q in y_nonzeroCols:
                gradVect0 = (-1) * (sigmoid_y * usr_rep).reshape(self.ITEM_FIELDS_NUM,1)
                WqGrads += gradVect0  # Indep of which q ( if in y's cols )
    
            # iterate over negs
            negCoef = 0.0
            for ind, y_negNonzeroCols in enumerate(y_negsNonzeroCols):
                if q in y_negNonzeroCols:
                    negCoef += sigmoids_negs[ind]
            gradVect1 = (negCoef * usr_rep.transpose()).reshape(self.ITEM_FIELDS_NUM,1)
            WqGrads += gradVect1  # Indep of which q ( if in y_neg's cols )
    
            gradsOfW[:,[q]] = WqGrads
        return gradsOfW

    # Return gradsOfV w.r.t. the user
    def getGradsOfV(self, V, itemsIndx, sumedW_y, sigmoid_y, sigmoidedSumedW):
        gradsOfV = {}
        itemLens = len(itemsIndx) 
        for itemInd in itemsIndx:
            gradVect3 = (-1) * (1.0 / itemLens) * sigmoid_y * sumedW_y  # Indep of which item
            gradVect4 = ((1.0 / itemLens) * sigmoidedSumedW).transpose()  # Indep of which item
            gradVect5 = 2 * self.LAMBDA * V[itemInd,:]
            gradsOfV[itemInd] = (gradVect3 + gradVect4 + gradVect5).reshape(self.ITEM_FIELDS_NUM,)  # Reshape to vect (not 2D array)
    
        return gradsOfV

    # gradsOfW 2-dim ndarray
    # gradsOfV: {
    #   someitemInd: itsGrad
    # ... }
    def updateByGradients(self, W, V, gradsOfW, gradsOfV, incrInd):
        scale = self.MOMENTUM if incrInd else 1/self.MOMENTUM
        W -= scale * self.LEARNING_RATE * gradsOfW
        for itemInd in gradsOfV:
            V[itemInd, :] -= scale * self.LEARNING_RATE * gradsOfV[itemInd]
        return W, V


    def predictLabels(self, usr_rep, W, bds):
        bestCols = map(lambda v: v[0], bds)
        bestScore = usr_rep.transpose().dot( sumOverW(W, bestCols) )
        bdsRanges = [range(bd[0], bd[1] + 1) for bd in bds]
        allCombs = list(product( *bdsRanges ))
        for comb in allCombs:
            y_nonzeroCols_sample = list(comb)
            sumedW = sumOverW(W, y_nonzeroCols_sample)
            score = usr_rep.transpose().dot( sumedW )
            if score >= bestScore:
                bestScore = score
                bestCols = y_nonzeroCols_sample
        return bestCols


    def getTerms(self, usrid, usr2labels, usr2NonzeroCols, usr2itemsIndx, W, usr_rep, usr2negsNonzeroCols):
        y                 = usr2labels[usrid]
        y_nonzeroCols     = usr2NonzeroCols[usrid]
        itemsIndx         = usr2itemsIndx[usrid]
        sumedW_y          = sumOverW(W, y_nonzeroCols)
        sigmoid_y         = sigmoid((-1) * usr_rep.transpose().dot(sumedW_y)) # equivalent to 1 - sigmoid(...)
        y_negsNonzeroCols = usr2negsNonzeroCols[usrid]
        sumedW_negs       = map(lambda y_negNonzeroCols: sumOverW(W, y_negNonzeroCols).reshape(self.ITEM_FIELDS_NUM,1), y_negsNonzeroCols)
        sigmoids_negs     = map(lambda sumedW_neg: sigmoid(usr_rep.transpose().dot(sumedW_neg)), sumedW_negs)
        sigmoidedSumedW   = np.zeros((self.ITEM_FIELDS_NUM,1))
        for ind, sigmoid_neg in enumerate(sigmoids_negs):
            sigmoidedSumedW += sigmoid_neg * sumedW_negs[ind]
        return y, y_nonzeroCols, itemsIndx, sumedW_y, sigmoid_y, \
            y_negsNonzeroCols, sumedW_negs, sigmoids_negs, sigmoidedSumedW
 
