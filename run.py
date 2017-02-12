import dataloaders.movielens100k as dataloader_movielens100k
import embedders.random_embedder as random_embedder
import poolers.sample_pooler as sample_pooler
from utils import get_labels, load_data
from config import USR_TOTAL_LABELS_FIELDS, MAX_TRAIN_NUM

import numpy as np
np.random.seed(123)  # Reproducibility

def checkConv(diff):
    return diff < 0.01

# TODO: modify this: hard code num of fields + each field's size(category num)
# ref: http://stackoverflow.com/questions/8386675/extracting-specific-columns-in-numpy-array
fieldNumList = [3,4,5]
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
    

def main():
    # load data as {
    #     usr_i(int): [ item_i0(int), item_i1(int), ... ]
    #     ... 
    # }
    dataloader = dataloader_movielens100k.dataloader_movielens100k()
    usr2items = dataloader.load('data/u.template.BOI')

    # embed data as: {
    #     usr_i: [{ 
    #             item_id: item_i0,
    #             features: embeded features for item_i0 (list of floats),
    #         },{
    #         ...}, {
    #             item_id: item_in,
    #             features: embeded features for item_in,
    #         }
    #     ], ...
    # }
    # init V_dict
    # TODO: val of V_dict should be truned to numpu's array
    embedder = random_embedder.random_embedder()
    usr2itemsfeatures, V_dict = embedder.embed_all(usr2items)

    # pool data as: {
    #     usr_i: usr_i's representation list,
    #     ....
    # }
    pooler = sample_pooler.sample_pooler()
    usr2representation = pooler.pool_all_inits(usr2itemsfeatures)

    # acquire usr labels as: {
    #     usr_i: demographic labels (list of list(could be empty one)),
    #     ...
    # }
    usr2labels = get_labels('data/u.template.labels')

    # assemble y_train as: [
    #     usr_i's labels,
    #     ...
    # ]; 
    y_train, usrids, usrid2yInd = load_data(usr2labels, usr2representation)
    y_train = np.array(y_train)
    print '[info] y == ', y_train, y_train.shape

    # init W
    W = np.random.rand(??usr_repredim, USR_TOTAL_LABELS_FIELDS)
    print '[info] W == ', W

    # learning for W, V (V_dict)
    # Rmk, since no significant row over col (or the reverse); just 
    #   implement the most naive way
    # ref: http://stackoverflow.com/questions/17954990/performance-of-row-vs-column-operations-in-numpy
    conv = False
    t = 0
    while not conv or t <= MAX_TRAIN_NUM:
        t += 1
        for usrid in usrids:
            # pooling
            usr_rep = pooler.pool_all(usr2items[usrid], V_dict)

            # get y, Wc, sumedW, negCoef
            y = y_train[ usrid2yInd[usr] ]
            Wc = takeCompounded(W, y) 
            sumedW = sumOverW(W, y)
            grads = map(lambda y_neg: sigmoid(usr_rep.transpose().dot( sumOverW(W, y_neg) )))
            negCoef = reduce(lambda u, v: u+v, grads)

            # TODO: implement negative sampling -- sampling thru emperialistic(train) data
            # TODO: question: should neg sampling be done whenever training for each epoch?
            ys_neg = negativeSampling()

            # get gradient of Wq (i.e. q-th column of W)
            gradsOfW = []
            for i in range(W.shape[1]):
                # if yq one 
                gradVect0 = 0.0
                gradVect1 = 0.0
                if i in y:
                    gradVect0 = sigmoid(-1 * usr_rep.transpose().dot( sumedW )) * usr_rep
                    gradVect1 = negCoef * (-1 * usr_rep)

                # TODO: decide when to update (over each t together?) 
                # Rmk: should be 2 * LAMBDA * ... (but just tune LAMBDA instead)
                gradVect2 = LAMBDA * W[:,[i]]
                gradsOfW.append( gradVect0 + gradVect1 + gradVect2 )

           # get gradient of Vp, p in X(i) 
           # TODO: only implemnet average poooling for now
           itemLensInverse = 1.0 / len(usr2items) 
           for i in usr2items:
               gradVect4 = sigmoid(-1 * usr_rep.transpose().dot() ) * itemLensInverse * sumedW
               gradVect5 = negCoef * (-1 * itemLensInverse) * sumedW


        # TODO: diff should route over tests
        diff = 0.0
        conv = checkConv(diff)

if __name__ == '__main__':
    main()
