import sys
sys.path.insert(0, '../')
import unittest
import numpy as np

from run import getRL, getOneError, getMicroF1ByCol, predictLabels, updateByGradients, getGradsOfW, getGradsOfV, sumOverW, sigmoid 
import poolers.sample_pooler as sample_pooler
from config import LAMBDA, LEARNING_RATE 

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
        sigmoids_negs = [ -sigmoid(10), -sigmoid(8) ] # assume yneg = [0,1,0,1], [1,0,0,1] resp
        y_negsNonzeroCols = [ [1,3], [0,3] ]

        gradsOfW = np.zeros((2,4))
        # q = 0
        grad0 = - sigmoid( -4 ) * usr_rep
        grad1 = - ( 0 + sigmoids_negs[1] ) * usr_rep 
        grad2 = 2 * LAMBDA * W[:,[0]]
        gradsOfW[:,[0]] = grad0.reshape(2,1) + grad1.reshape(2,1) + grad2
        #print grad0, grad1, grad2

        # q = 1
        grad0 = - sigmoid( -4 ) * usr_rep * 0
        grad1 = - ( sigmoids_negs[0] + 0 ) * usr_rep 
        grad2 = 2 * LAMBDA * W[:,[1]]
        gradsOfW[:,[1]] = grad0.reshape(2,1) + grad1.reshape(2,1) + grad2
        #print grad0, grad1, grad2

        # q = 2
        grad0 = - sigmoid( -4 ) * usr_rep  
        grad1 = - ( 0 + 0 ) * usr_rep 
        grad2 = 2 * LAMBDA * W[:,[2]]
        gradsOfW[:,[2]] = grad0.reshape(2,1) + grad1.reshape(2,1) + grad2
        #print grad0, grad1, grad2

        # q = 3
        grad0 = - sigmoid( -4 ) * usr_rep * 0
        grad1 = - ( sigmoids_negs[0] + sigmoids_negs[1] ) * usr_rep 
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
        grad4 = (-1) * sigmoidedSumedW.transpose() * (1.0 / 2)
        grad5 = 2 * LAMBDA * V[0]
        gradsOfV[0] = (grad3 + grad4 + grad5).reshape(2, )

        # V[1]
        grad3 = (-1) * sigmoid_y * (1.0 / 2) * sumedW_y
        grad4 = (-1) * sigmoidedSumedW.transpose() * (1.0 / 2)
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
if __name__ == '__main__':
    unittest.main()
