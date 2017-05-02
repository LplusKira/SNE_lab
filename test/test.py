import sys, math
sys.path.insert(0, '../')
import unittest
import numpy as np

from run import getRL, getOneError, getMicroF1ByCol, getCoverage, getAvgPrecision, predictLabels, updateByGradients, getGradsOfW, getGradsOfV, getTerms, sumOverW, sigmoid 
import poolers.sample_pooler as sample_pooler
from config import LAMBDA, LEARNING_RATE, ITEM_FIELDS_NUM
from utils import getDistribution, negativeSample

class TestRun(unittest.TestCase):
    # validate pooler 
    def test_pooler(self):
        pooler = sample_pooler.sample_pooler()
        usr2itemsIndx = {
          1: [0,3,4],
        }
        V = np.array( [ [ 1.0, 1.0 ], \
                        [ 2.0, 2.0 ], \
                        [ 2.0, 2.0 ], \
                        [ 2.0, 2.0 ], \
                        [ 3.0, 3.0 ], \
                      ])
        usr_rep = pooler.pool_all(usr2itemsIndx[1], V)
        for x, e in enumerate(usr_rep):
            self.assertEqual(usr_rep[x], 2.0)

    # validate sumOverW
    def test_sumOverW(self):
        W = np.array( [ [ 0, 1, 2, 4 ], \
                        [ 0, 1, 2, 4 ], \
                      ])
        y_nonzeroCols = [0, 2]
        sumedW_y = sumOverW(W, y_nonzeroCols)
        for x, ele in enumerate(sumedW_y):
            self.assertEqual(ele, 2.0)

    # validate the terms get back
    def test_getTerms(self):
        usrid = 0
        usr2labels = {
          0: [0, 1, 0, 1],
          1: [1, 0, 1, 0],
        }
        usr2NonzeroCols = {
          0: [1, 3], 
          1: [0, 2],
        }
        usr2itemsIndx = {
          0: [0, 1],
          1: [2, 3],
        } 
        W = np.array( [ [ 0, 1, 2, 4 ], \
                        [ 0, 1, 2, 4 ], \
                      ])
        usr_rep         = np.array( [ [ 1, 1 ], \
                                      [2, 2] , \
                                     ])[0]  # should be [1,1]'
        usr2negsNonzeroCols = {
          0: [[0, 3], [1, 2]],
          1: [[1, 3], [0, 2]],
        } 

        # actual:
        y, y_nonzeroCols, itemsIndx, sumedW_y, sigmoid_y, y_negsNonzeroCols, sumedW_negs, sigmoids_negs, sigmoidedSumedW = getTerms(usrid, usr2labels, usr2NonzeroCols, usr2itemsIndx, W, usr_rep, usr2negsNonzeroCols)
        # expected:
        y2 = [0, 1, 0, 1]
        y_nonzeroCols2 = [1,3]
        itemsIndx2 = [0, 1]
        sumedW_y2 = sumOverW(W, y_nonzeroCols2)
        sigmoid_y2 = sigmoid(- usr_rep.transpose().dot(sumedW_y2))
        y_negsNonzeroCols2 = [[0, 3], [1, 2]]
        sumedW_negs2 = [ sumOverW(W, [0, 3]).reshape(ITEM_FIELDS_NUM,1), sumOverW(W, [1, 2]).reshape(ITEM_FIELDS_NUM,1) ]
        sigmoids_negs2 = [ sigmoid( usr_rep.transpose().dot(sumedW_negs2[0]) ), sigmoid( usr_rep.transpose().dot(sumedW_negs2[1]) ) ]
        sigmoidedSumedW2 = sigmoids_negs2[0] * sumedW_negs2[0] + sigmoids_negs2[1] * sumedW_negs2[1]

        self.assertEqual(y2, y)
        self.assertEqual(y_nonzeroCols2, y_nonzeroCols)
        self.assertEqual(itemsIndx2, itemsIndx)
        for x, e in enumerate(sumedW_y2):
            self.assertEqual(sumedW_y2[x], sumedW_y[x])
        self.assertEqual(sigmoid_y2, sigmoid_y)
        self.assertEqual(y_negsNonzeroCols2, y_negsNonzeroCols)
        for ind, e in enumerate(sumedW_negs2):
            for x, e2 in enumerate(e):
                self.assertEqual(e[x], sumedW_negs[ind][x])
        self.assertEqual(sigmoids_negs2, sigmoids_negs)
        for x, e in enumerate(sigmoidedSumedW2):
            self.assertEqual(sigmoidedSumedW2[x], sigmoidedSumedW[x])


    # validate the gradient of W, by user
    def test_getGradsOfW(self):
        W = np.array( [ [ 0, 1, 2, 4 ], \
                        [ 0, 1, 2, 4 ], \
                      ])
        y_nonzeroCols = [0, 2]                  # assume y = [1,0,1,0]
        sumedW_y = sumOverW(W, y_nonzeroCols)   # should be [2,2]'
        usr_rep = np.array( [ [ 1, 1 ], \
                              [2, 2] , \
                             ])[0]              # should be [1,1]'
        sigmoid_y = sigmoid( - usr_rep.transpose().dot(sumedW_y) )
        sigmoids_negs = [ sigmoid(10), sigmoid(8) ] # assume yneg = [0,1,0,1], [1,0,0,1] resp
        y_negsNonzeroCols = [ [1,3], [0,3] ]

        gradsOfW = np.zeros((2,4))
        # q = 0
        grad0 = - sigmoid( -4 ) * usr_rep
        grad1 = ( 0 + sigmoids_negs[1] ) * usr_rep 
        grad2 = 2 * LAMBDA * W[:,[0]]
        gradsOfW[:,[0]] = grad0.reshape(2,1) + grad1.reshape(2,1) + grad2
        #print grad0, grad1, grad2

        # q = 1
        grad0 = - sigmoid( -4 ) * usr_rep * 0
        grad1 = ( sigmoids_negs[0] + 0 ) * usr_rep 
        grad2 = 2 * LAMBDA * W[:,[1]]
        gradsOfW[:,[1]] = grad0.reshape(2,1) + grad1.reshape(2,1) + grad2
        #print grad0, grad1, grad2

        # q = 2
        grad0 = - sigmoid( -4 ) * usr_rep  
        grad1 = ( 0 + 0 ) * usr_rep 
        grad2 = 2 * LAMBDA * W[:,[2]]
        gradsOfW[:,[2]] = grad0.reshape(2,1) + grad1.reshape(2,1) + grad2
        #print grad0, grad1, grad2

        # q = 3
        grad0 = - sigmoid( -4 ) * usr_rep * 0
        grad1 = ( sigmoids_negs[0] + sigmoids_negs[1] ) * usr_rep 
        grad2 = 2 * LAMBDA * W[:,[3]]
        gradsOfW[:,[3]] = grad0.reshape(2,1) + grad1.reshape(2,1) + grad2
        #print grad0, grad1, grad2

        actualGradsOfW = getGradsOfW(W, y_nonzeroCols, sigmoid_y, usr_rep, sigmoids_negs, y_negsNonzeroCols)
        expectGradsOfW = gradsOfW
        for (x, y), ele in np.ndenumerate(actualGradsOfW):
            self.assertEqual(actualGradsOfW[x,y], expectGradsOfW[x,y])
        
    
    # validate the gradient of W, by user
    def test_getGradsOfV(self):
        W               = np.array( [ [ 0, 1, 2, 4 ], \
                                      [ 0, 1, 2, 4 ], \
                                    ])
        V               = np.array( [ [1, 1], \
                                     [2, 2], \
                                     [3, 3], \
                                     [4, 4], \
                                   ])
        itemsIndx       = [0,3]
        y_nonzeroCols   = [0, 2]                       # assume y = [1,0,1,0]
        sumedW_y        = sumOverW(W, y_nonzeroCols)   # should be [2,2]'
        usr_rep         = np.array( [ [ 1, 1 ], \
                                      [2, 2] , \
                                     ])[0]              # should be [1,1]'
        sigmoid_y       = sigmoid( - usr_rep.transpose().dot(sumedW_y) )
        sigmoidedSumedW = np.array( [ [1], \
                                      [1], \
                                  ])

        gradsOfV = {}
        # V[0]
        grad3 = (-1) * sigmoid_y * (1.0 / 2) * sumedW_y
        grad4 = sigmoidedSumedW.transpose() * (1.0 / 2)
        grad5 = 2 * LAMBDA * V[0]
        gradsOfV[0] = (grad3 + grad4 + grad5).reshape(2, )

        # V[1]
        grad3 = (-1) * sigmoid_y * (1.0 / 2) * sumedW_y
        grad4 = sigmoidedSumedW.transpose() * (1.0 / 2)
        grad5 = 2 * LAMBDA * V[3]
        gradsOfV[3] = (grad3 + grad4 + grad5).reshape(2, )

        actualGradsOfV = getGradsOfV(V, itemsIndx, sumedW_y, sigmoid_y, sigmoidedSumedW)
        expectGradsOfV = gradsOfV
        for itemIndx in actualGradsOfV:
            for x, e in enumerate(actualGradsOfV[itemIndx]):
                self.assertEqual(actualGradsOfV[itemIndx][x], expectGradsOfV[itemIndx][x])

    # validate updateByGradients
    def test_updateByGradients(self):
        W               = 1.0 * np.array( [ [ 0, 1, 2, 4 ], \
                                      [ 0, 1, 2, 4 ], \
                                    ])
        V               = 1.0 * np.array( [ [1, 1], \
                                     [2, 2], \
                                     [3, 3], \
                                     [4, 4], \
                                   ])
        gradsOfW        = 10000 * np.array( [ [ 1, 1, 1, 1 ], \
                                             [ 1, 1, 1, 1 ], \
                                           ])
        gradsOfV        = {
            0: 10000 * np.array([1, 1]).reshape(2, ),
            3: 10000 * np.array([2, 1]).reshape(2, ),
        }
        incrInd         = True
     
        expectW = np.array([ [-2, -1, 0, 2], \
                             [-2, -1, 0, 2], \
                           ])
        expectV = np.array([ [-1, -1], \
                             [2, 2], \
                             [3, 3], \
                             [0, 2], \
                            ])
        
        actualW, actualV = updateByGradients(W, V, gradsOfW, gradsOfV, incrInd)
        for (x, y), ele in np.ndenumerate(actualV):
            self.assertEqual(actualV[x,y], expectV[x,y])

        for (x, y), ele in np.ndenumerate(actualW):
            self.assertEqual(actualW[x,y], expectW[x,y])

    # validate predictLabels
    def test_predictLabels(self):
        usr_rep = np.array([1, 1, 1]).reshape(3, )
        W       = np.array( [ [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                              # 0,       3, 4, 5,            ... 11 ...                                       26
                             ])
        expectBestCols = [3, 5, 11]
        actualBestCols = predictLabels(usr_rep, W)
        for ind, ele in enumerate(expectBestCols):
            self.assertEqual(actualBestCols[ind], expectBestCols[ind])

    # validate getMicroF1ByCol
    def test_getMicroF1ByCol(self):
        pooler = sample_pooler.sample_pooler()
        V       = np.array([ [1, 1, 1], \
                             [1, 1, 1], \
                           ])
        W       = np.array( [ [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                              # 0,       3, 4, 5,            ... 11 ...                                       26
                             ])
        usr2itemsIndx = {
          0: [0,1],
          1: [1],
          2: [0],
        }

        usr2NonzeroCols = {
          0: [0, 4, 11],
          1: [1, 5, 10],
          2: [3, 4, 26],
        }

        expectMicroF1 = 1.0 / 3
        actualMicroF1 = getMicroF1ByCol(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        self.assertEqual(expectMicroF1, actualMicroF1)
        
    # validate getOneError
    def test_getOneError(self):
        pooler = sample_pooler.sample_pooler()
        V       = np.array([ [1, 1, 1], \
                             [1, 1, 1], \
                           ])
        W       = np.array( [ [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                              # 0,       3, 4, 5,            ... 11 ...                                       26
                             ])
        usr2itemsIndx = {
          0: [0,1],
          1: [1],
          2: [0],
        }

        usr2NonzeroCols = {
          0: [0, 4, 11],
          1: [1, 4, 10],
          2: [2, 4, 26],
        }

        # should all guess 3,5,11
        expectOneError = 2/3.0
        actualOneError = getOneError(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        self.assertEqual(expectOneError, actualOneError)
    
    # validate getRL
    def test_getRL(self):
        pooler = sample_pooler.sample_pooler()
        V       = np.array([ [1, 1, 1], \
                             [1, 1, 1], \
                           ])
        W       = np.array( [ [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                              # 0,       3, 4, 5,            ... 11 ...                                       26
                             ])
        usr2itemsIndx = {
          0: [0,1],
          1: [1],
        }

        usr2NonzeroCols = {
          0: [3, 5, 11],  # --> should be 0.0
          1: [1, 4, 10],  # --> should be 18/72.0
        }
       
        # should all guess 3,5,11
        expectRL = (0.0 + 1/4.0) / 2
        actualRL = getRL(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        self.assertEqual(expectRL, actualRL)

    # validate getDistribution
    def test_getDistribution(self):
        usr2labels = {
            0: [0, 0, 1, 1, 0],
            1: [0, 0, 1, 1, 0],
            2: [0, 1, 0, 1, 0],
            3: [0, 1, 0, 0, 1],
        }
        actualCdfByLabels, actualLabelsList = getDistribution(usr2labels)
        expectCdfByLabels = [ 0.5,            0.75,            1 ] 
        expectLabelsList =  [[0, 0, 1, 1, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0]]
        self.assertEqual(len(actualCdfByLabels), len(expectCdfByLabels))
        for ind, ele in enumerate(actualCdfByLabels):
            self.assertEqual(actualCdfByLabels[ind], expectCdfByLabels[ind])
        for ind, ele in enumerate(actualLabelsList):
            self.assertEqual(str(actualLabelsList[ind]), str(expectLabelsList[ind]))

    # validate negative sample
    def test_negativeSample(self):
        usr2labels = {
            0: [0, 0, 1, 1, 0],
            1: [0, 0, 1, 1, 0],
            2: [0, 1, 0, 1, 0],
            3: [0, 1, 0, 0, 1],
        }  # <-- actually only use the usrs in it here
        cdfByLabels = [ 0.5,            0.75,            1 ]
        labelsList =  [[0, 0, 1, 1, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0]]

        # assert ordery
        usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList, k=100)
        for usr in usr2NegativeSamples:
            for ind, sample in enumerate(usr2NegativeSamples[usr]):
                for col in range(len(sample)):
                    if col in usr2negsNonzeroCols[usr][ind]:
                        self.assertEqual(sample[col], 1)
                    else:
                        self.assertEqual(sample[col], 0)

        # to each usr
        #   should have about 500 for [0, 0, 1, 1, 0]
        #   should have about 250 for [0, 1, 0, 0, 1]
        #   should have about 250 for [0, 1, 0, 1, 0]
        # to usr 0, 1
        #   should have about 500 for [0, 1, 0, 1, 0]
        #   should have about 500 for [0, 1, 0, 0, 1]
        # to usr 2
        #   should have about 666 for [0, 0, 1, 1, 0]
        #   should have about 334 for [0, 1, 0, 0, 1]
        # to usr 3
        #   should have about 666 for [0, 0, 1, 1, 0]
        #   should have about 334 for [0, 1, 0, 1, 0]
        usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList, k=1000)
        expectDict = {
            0: {
            str([0, 0, 1, 1, 0]): 0,
            str([0, 1, 0, 0, 1]): 500, 
            str([0, 1, 0, 1, 0]): 500,
        }, 1: {
            str([0, 0, 1, 1, 0]): 0,
            str([0, 1, 0, 0, 1]): 500, 
            str([0, 1, 0, 1, 0]): 500,
        }, 2: {
            str([0, 0, 1, 1, 0]): 666,
            str([0, 1, 0, 1, 0]): 0,
            str([0, 1, 0, 0, 1]): 334, 
        }, 3: {
            str([0, 0, 1, 1, 0]): 666,
            str([0, 1, 0, 1, 0]): 334,
            str([0, 1, 0, 0, 1]): 0, 
        }}

        for usr in usr2NegativeSamples:
            actualDict = {
                str([0, 0, 1, 1, 0]): 0,
                str([0, 1, 0, 0, 1]): 0, 
                str([0, 1, 0, 1, 0]): 0,
            }
            for sample in usr2NegativeSamples[usr]:
                actualDict[str(sample)] += 1

            for comb in actualDict:
                self.assertEqual(math.fabs(actualDict[comb] - expectDict[usr][comb]) < 40, True)

    # validate getCoverage
    def test_getCoverage(self):
        pooler = sample_pooler.sample_pooler()
        V       = np.array([ [1, 1, 1], \
                             [1, 1, 1], \
                           ])
        W       = np.array( [ [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                              # 0,       3, 4, 5,            ... 11 ...                                       26
                             ])
        usr2itemsIndx = {
          0: [0,1],
          1: [1],
          2: [0],
        }

        usr2NonzeroCols = {
          0: [0, 4, 11],
          1: [1, 4, 10],
          2: [2, 4, 26],
        }

        # should all guess 3,5,11
        expectCoverage = (6.0/27 + 11.0/27 + 26.0/27) / 3
        actualOneError = getCoverage(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        self.assertEqual(expectCoverage, actualOneError) 

    # validate getAvgPrecision
    def test_getAvgPrecision(self):
        pooler = sample_pooler.sample_pooler()
        V       = np.array([ [1, 1, 1], \
                             [1, 1, 1], \
                           ])
        W       = np.array( [ [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                              # 0,       3, 4, 5,            ... 11 ...                                       26
                             ])
        usr2itemsIndx = {
          0: [0,1],
          1: [1],
          2: [0],
        }

        usr2NonzeroCols = {
          0: [0, 4, 11],
          1: [1, 4, 10],
          2: [2, 4, 26],
        }

        # should all guess 3,5,11
        expectAvgPrecision = ( ( 2.0 / 4 + 3.0 / 7 + 1.0 / 3) / 3 + ( 1.0/5 + 2.0/7 + 3.0/12) / 3 + ( 1.0/6 + 2.0/7 + 3.0/27) / 3 ) / 3
        actualAvgPrecision = getAvgPrecision(W, V, usr2itemsIndx, usr2NonzeroCols, pooler)
        self.assertEqual(expectAvgPrecision, actualAvgPrecision)

if __name__ == '__main__':
    unittest.main()
