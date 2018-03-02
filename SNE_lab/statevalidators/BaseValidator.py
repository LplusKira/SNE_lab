from time import time
from math import fabs
import logging
from sys import path
path.append('../')
from config import TEST_SNE, TEST_DIR


''' By-fold state handler '''
class BaseValidator(object):
    MF1Bd = 1.0
    LossDiffBd = 1e-6
    LossDiffNegBd = LossDiffBd * 10
    EnoughRuns = 100
    NegSampleFreq = 100
    RevealRun = 10
    CalLossRun = 1000
    RootDir = '../report/' if not TEST_SNE else '../' + TEST_DIR

    def __init__(self, dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, predictTrain=False, write2File=True):
        self.dataset = dataset
        self.datasetSub = datasetSub
        self.curFold = curFold
        self.totalFolds = totalFolds
        self.usr2itemsIndxTrain = usr2itemsIndxTrain
        self.usr2itemsIndxValid = usr2itemsIndxValid
        self.predictTrain = predictTrain
        self.write2File = write2File
        self.MAX_TRAIN_NUM = MAX_TRAIN_NUM
        self.NegSampleRun = self.MAX_TRAIN_NUM * 0.9
        self.ITEM_FIELDS_NUM = ITEM_FIELDS_NUM
        self.createFile = False
        self.initTime = time()  # In secs

        # Stats for conv
        self.t = 0

        # Stats for debugging
        self.loss = float('inf')
        self.prevLoss = float('inf')
        self.lossDiff = float('inf')

        # Optimization configs
        self.incrInd = False

    def logFoldInfo(self):
        logging.info('(curFold, totalFolds) = ' + str(self.curFold) + ',' + str(self.totalFolds))
        logging.info('usrs in train: ' + str(len(self.usr2itemsIndxTrain)))
        logging.info('usrs in valid: ' + str(len(self.usr2itemsIndxValid)))

    def logStartNegSample(self):
        logging.info('resample usr negative samples')

    def logNegSampleInfo(self, usr2NegativeSamples):
        logging.info('diff' + str(self.lossDiff))
        logging.info('usr2NegativeSamples: ')
        # XXX formatted output, perhaps?
        for usrid in usr2NegativeSamples:
            print usrid
            for sample in usr2NegativeSamples[usrid]:
                print ' ', sample

    def logLossStates(self, W, V, loss):
        logging.info('loss (only for train -- since this requires samples) == ' + str(loss))
        logging.info('loss diff == ' + str(self.lossDiff))
        logging.debug('W[:,0] == ' + str(W[:, 0]))
        logging.debug('V == ' + str(V))

    def logStartPrediction(self):
        logging.info('start predicting')
        if not self.predictTrain:
            logging.info('will NOT predict on train data')

    def logCollectingStats(self):
        logging.info('Start collecting stats')

    def logStats(self, data):
        name = data['name']
        KPIs = data['KPIs']
        for kpi, val in KPIs.iteritems():
            s = ' '.join([name, 'data', kpi, '==', str(val)])
            logging.info(s)

    def writeCSVStats(self, data):
        # Format: dataset, latentVarsNum, fold, curRun, curFoldspentTime, lossDiff, train/valid, metric, val
        dataset = self.dataset + self.datasetSub if self.datasetSub else self.dataset
        latentVarsNum = self.ITEM_FIELDS_NUM
        fold = self.curFold
        curRun = self.t
        curFoldspentTime = int(time() - self.initTime)  # In secs
        lossDiff = self.lossDiff if self.shouldCalLoss() else None
        splitDataName = data['name']
        baseCols = [dataset, latentVarsNum, fold, curRun,
                    curFoldspentTime, lossDiff, splitDataName]
        baseCols = [str(ele) for ele in baseCols]
        KPIs = data['KPIs']

        outFile = self.RootDir + ''.join([str(latentVarsNum), 'F', dataset])
        logging.info('write to' + outFile)
        flag = 'a' if self.curFold != 0 or self.createFile else 'w+'  # Overwrite at first fold
        self.createFile = True
        with open(outFile, flag) as f:
            for kpi, val in KPIs.iteritems():
                cols = baseCols[:]
                cols += [kpi, str(val)]
                line = ','.join(cols) + '\n'
                f.write(line)

    def logRealPredictedVals(self, data):
        usr2itemsIndx = data['usr2itemsIndx']
        usr2NonzeroCols = data['usr2NonzeroCols']
        u2predictions = data['u2predictions']
        name = data['name']
        logging.info(' '.join(['for', name, 'data, print real vals & predicted vals ... ']))
        logging.info('usrid, actual, predicted')
        for usrid in usr2itemsIndx:
            y_nonzeroCols = usr2NonzeroCols[usrid]
            bestCols = u2predictions[usrid]
            logging.info(' '.join([str(usrid), str(y_nonzeroCols), str(bestCols)]))

    # States-related funcs
    def nextRun(self):
        self.t += 1
        logging.info('run == ' + str(self.t))

    def notConv(self):
        # MicroF1 btw [0,1]; bigger => higer precision, higer recall
        return fabs(self.lossDiff) > self.LossDiffBd and self.t < self.MAX_TRAIN_NUM

    def shouldNegSample(self):
        # Cond1: many runs and long way to go
        # Cond2: |loss' diff| small (perhaps local min)
        return (self.t >= self.EnoughRuns and self.t == self.NegSampleRun) or \
               (fabs(self.lossDiff) < self.LossDiffNegBd and self.t % self.NegSampleFreq == 0)

    def shouldRevealStats(self):
        return self.t % self.RevealRun == 0

    def shouldCalLoss(self):
        return self.t % self.CalLossRun == 0

    def updateLossState(self, loss):
        self.loss = loss
        self.lossDiff = self.loss - self.prevLoss
        self.prevLoss = self.loss

    def shouldPredictTrain(self):
        return self.predictTrain

    def getDataStats(self, usr2itemsIndxValid, usr2itemsIndxTrain, usr2NonzeroCols):
        dataStats = [{
            'name': 'valid',
            'usr2itemsIndx': usr2itemsIndxValid,
            'u2predictions': {},
            'usr2NonzeroCols': usr2NonzeroCols,
            'KPIs': [],
        }]
        if self.shouldPredictTrain():
            dataStats.append({
                'name': 'train',
                'usr2itemsIndx': usr2itemsIndxTrain,
                'u2predictions': {},
                'usr2NonzeroCols': usr2NonzeroCols,
                'KPIs': [],
            })
        return dataStats
