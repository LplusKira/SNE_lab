from dataloaders.EgoNetwork import ENLoader
from dataloaders.Youtube import YTLoader
from dataloaders.MovieLens100K import ML100KLoader
from dataloaders.MovieLens1M import ML1MLoader
DATA2LOADER = {  # By subdir in data/
    'ml-100k': ML100KLoader,
    #'yelp': XXX,
    'ml-1m': ML1MLoader,
    'ego-net': ENLoader,
    'youtube': YTLoader,
}
from poolers.sample_pooler import sample_pooler
#from statevalidators.Basevalidator import Basevalidator
from statevalidators.ENValidator import ENValidator
from statevalidators.YTValidator import YTValidator
DATA2VALIDATOR = {  # By subdir in data/
    #'ml-100k': XXX,
    #'yelp': XXX,
    #'ml-1m': XXX,
    'ego-net': ENValidator,
    'youtube': YTValidator,
}
from updator.Baseupdator import Baseupdator
from utils import getDistribution, negativeSample, splitKfolds, merge_two_dicts, getMicroF1ByCol, getOneError, getRL, getCoverage, getAvgPrecision, getHammingLoss
from config import LogFlags

import math, random, sys, traceback
from numpy import random as nprandom
from time import gmtime, strftime


def parseArgs(argv, **kwargs):
    def getSubDir(rating_file, usr2labels_file):
        rating_file_subdir = rating_file.split('/')[2]
        usr2labels_file_subdir = usr2labels_file.split('/')[2]
        if rating_file_subdir == usr2labels_file_subdir:
            return rating_file_subdir
        return None

    mainFile = argv[0]
    nprandom.seed(int(argv[1]))  # Reproducibility
    random.seed(int(argv[1]))  # Reproducibility
    foldNum = int(argv[2])
    dataset = argv[3]
    rating_file = kwargs.get('rating_file')  # By 'get', default None
    usr2labels_file = kwargs.get('usr2labels_file')
    subtitle = kwargs.get('sub')
         
    usage = LogFlags['USE'] + 'python -u ' + mainFile + \
        ' randomSeed(int) foldNum dataset subtitle(opt) ratingFile(opt) featureFile(opt)' + \
        '(ratingFile and featureFile only coexist under the same dir)'
    # Handle ratingFile, featureFile, dataset's conflicts
    if (not rating_file and usr2labels_file) or (not rating_file and usr2labels_file):
        raise Exception(usage)
    elif rating_file and usr2labels_file:
        subdir = getSubDir(rating_file, usr2labels_file)
        if not subdir:
            raise Exception(usage)
    return foldNum, dataset, subtitle, rating_file, usr2labels_file

def main(argv):
    ''' Parse args, init dataloader '''
    foldNum, dataset, subtitle, rating_file, usr2labels_file = parseArgs(argv[:4], **dict(arg.split('=') for arg in argv[4:]))
    if rating_file and usr2labels_file:
        dataloader = DATA2LOADER[dataset](
            rating_file = rating_file, 
            usr2labels_file = usr2labels_file,
            sub = subtitle,
        )
    else:
        dataloader = DATA2LOADER[dataset]()

    ''' Load training conifgs '''
    NEG_SAMPLE_NUM, \
    ITEM_FIELDS_NUM, \
    MAX_TRAIN_NUM, \
    LEARNING_RATE, \
    MOMENTUM, \
    LAMBDA = dataloader.getTrainingConf()

    ''' Load each usr's BOI (and for valid data) ''' 
    usr2itemsIndx, ind2itemNum = dataloader.load()
    usrs = map(lambda usr: usr, usr2itemsIndx)

    ''' Assert enough usrs '''
    if foldNum > len(usrs):
        s = LogFlags['ERR'] + ' '.join(['foldNum: ', str(foldNum), '>', 'usrNums:', str(usrs)])
        raise Exception(s)

    ''' Acquire (for all usrs) usr2labels & usr2NonzeroCols ''' 
    usr2labels = dataloader.get_labels(usrs)
    usr2NonzeroCols = dataloader.get_nonZeroCols()

    ''' Init Baseupdator '''
    baseupdator = Baseupdator(*dataloader.getTrainingConf())

    # XXX check: every op needed in fold
    ''' K-fold validation '''
    kfolds = splitKfolds(usr2itemsIndx, foldNum)
    for ind, fold in enumerate(kfolds):
        # Init folds
        usr2itemsIndxValid = fold
        # XXX too verbose, elestic way?
        usr2itemsIndxTrain = {}
        for tind, tfold in enumerate(kfolds):
            if ind != tind:
                usr2itemsIndxTrain = merge_two_dicts(usr2itemsIndxTrain, tfold)
        # Init statevalidator
        statevalidator = DATA2VALIDATOR[dataset](
            dataset = dataset,
            datasetSub = dataloader.getDataSub(),
            curFold = ind, 
            totalFolds = len(kfolds), 
            usr2itemsIndxTrain = usr2itemsIndxTrain, 
            usr2itemsIndxValid = usr2itemsIndxValid,
            MAX_TRAIN_NUM = MAX_TRAIN_NUM,
            ITEM_FIELDS_NUM = ITEM_FIELDS_NUM,
        )
        statevalidator.logFoldInfo()
        
        
        ''' acquire (k times) usr2NegativeSamples & usr2negsNonzeroCols ''' 
        cdfByLabels, labelsList = getDistribution(usr2labels)
        usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(usr2labels, cdfByLabels, labelsList, k=NEG_SAMPLE_NUM)
        statevalidator.__log__(LogFlags['INFO'] + 'usr2NegativeSamples, usr2negsNonzeroCols created')


        ''' init V to [-1, 1) ''' 
        numOfItems = len(ind2itemNum)
        V = 2 * nprandom.rand(numOfItems, ITEM_FIELDS_NUM) -1
        statevalidator.__log__(LogFlags['INFO'] + 'V inited, V.shape == ' + str(V.shape) + ' == (num items, itemFeatures length)')


        ''' init W to [-1, 1); init pooler'''
        # Warn: assume ITEM_FIELDS_NUM is the same after usr's representation's dimension
        totalLabelsNum = dataloader.gettotalLabelsNum()
        W = 2 * nprandom.rand(ITEM_FIELDS_NUM, totalLabelsNum) -1
        pooler = sample_pooler()
        statevalidator.__log__(LogFlags['INFO'] + 'W & pooler inited, W.shape == ' + str(W.shape) + ' == (itemFeatures length, total labels num)')
        statevalidator.debug('W', W)
        statevalidator.debug('V', V)


        ''' learning for W, V '''
        microF1Valid = 0.0  # <--- btw [0,1]; bigger => higer precision, higer recall
        while statevalidator.notConv():
            # Init next run
            statevalidator.nextRun()

            # NegSampling or not
            if statevalidator.shouldNegSample():
                statevalidator.logStartNegSample()
                usr2NegativeSamples, usr2negsNonzeroCols = negativeSample(
                    usr2labels, 
                    cdfByLabels, 
                    labelsList, 
                    k=NEG_SAMPLE_NUM
                )
                statevalidator.logNegSampleInfo(usr2NegativeSamples)

            for usrid in usr2itemsIndxTrain:
                # Pooling
                usr_rep = pooler.pool_all(usr2itemsIndxTrain[usrid], V)

                # Get y, sumedW(for y AND negs), sigmoids(for y AND negs)
                # Warn: assume all usrs have all labels
                y, y_nonzeroCols, itemsIndx, sumedW_y, sigmoid_y, \
                y_negsNonzeroCols, sumedW_negs, sigmoids_negs, \
                sigmoidedSumedW = baseupdator.getTerms(
                    usrid, 
                    usr2labels, 
                    usr2NonzeroCols, 
                    usr2itemsIndxTrain, 
                    W, 
                    usr_rep, 
                    usr2negsNonzeroCols,
                )

                # Warn: update W by usr, not by epoch 
                # Get gradient of Wq (i.e. q-th column of W)
                gradsOfW = baseupdator.getGradsOfW(
                    W, 
                    y_nonzeroCols, 
                    sigmoid_y, 
                    usr_rep, 
                    sigmoids_negs, 
                    y_negsNonzeroCols,
                )
                
                # Warn: update V by usr, not by epoch 
                # Get gradient of Vitem, itemInd in X(usrid) 
                # Warn: only average poooling for now
                #   gradVect3, and gradVect4 happen to hold the same values over all items in this case
                gradsOfV = baseupdator.getGradsOfV(
                    V, 
                    itemsIndx, 
                    sumedW_y, 
                    sigmoid_y, 
                    sigmoidedSumedW,
                )

                # Update gradients to W, V
                W, V = baseupdator.updateByGradients(
                    W, 
                    V, 
                    gradsOfW, 
                    gradsOfV, 
                    statevalidator.incrInd,
                )

            # Time to reveal stats/predictions
            if statevalidator.shouldRevealStats():
                # Cal loss if needed
                if statevalidator.shouldCalLoss():
                    avgloss = baseupdator.getAvgLoss(
                        W, 
                        V, 
                        usr2NonzeroCols, 
                        usr2negsNonzeroCols, 
                        usr2itemsIndxTrain, 
                        pooler,
                    )
                    statevalidator.updateLossState(avgloss)
                    statevalidator.logLossStates(W, V, avgloss)

                # Do predictions
                statevalidator.logStartPrediction()
                dataStats = statevalidator.getDataStats(
                    usr2itemsIndxValid, 
                    usr2itemsIndxTrain,
                    usr2NonzeroCols
                )
                for d in dataStats:
                    usr2itemsIndx = d['usr2itemsIndx']
                    u2predictions = d['u2predictions']
                    for usrid in usr2itemsIndx:
                        usr_rep = pooler.pool_all(usr2itemsIndx[usrid], V)
                        bestCols = baseupdator.predictLabels(usr_rep, W, dataloader.getBds())
                        u2predictions[usrid] = bestCols

                # Collect Stats
                statevalidator.logCollectingStats()
                KPI2getters = {
                    'microF1': getMicroF1ByCol,
                    'oneError': getOneError,
                    'RL': getRL,
                    'coverage': getCoverage,
                    'avgPrec': getAvgPrecision,
                    'hammingLoss': getHammingLoss,
                }
                for d in dataStats:
                    KPIArgs = {
                        'W': W,
                        'V': V,
                        'usr2itemsIndx': d['usr2itemsIndx'],
                        'usr2NonzeroCols': usr2NonzeroCols,
                        'u2predictions': d['u2predictions'],
                        'totalLabelsNum': dataloader.gettotalLabelsNum(),
                        'rlPairsCnt': dataloader.getRLPairsCnt(),
                    }
                    d['KPIs'] = { kpi: getter(KPIArgs) for kpi, getter in KPI2getters.iteritems() }
                    #statevalidator.logStats(d)
                    statevalidator.writeCSVStats(d)

                # Log real, predicted
                for d in dataStats:
                    statevalidator.logRealPredictedVals(d)

if __name__ == '__main__':
    main(sys.argv[:])