from sys import path
path.append('../')
from time import strftime, gmtime, time
from config import LogFlags, DEBUG
from math import fabs

''' By-fold state handler '''
class BaseValidator(object):
    MF1Bd = 1.0
    LossDiffBd = 1e-6
    EnoughRuns = 100
    NegSampleFreq = 100
    RevealRun = 10
    CalLossRun = 1000
    initTime = time()  # In secs
    RootDir='../report/'

    def __init__(self, dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, silence=False, predictTrain=False, write2File=True):
        self.dataset = dataset
        self.datasetSub = datasetSub
        self.curFold = curFold
        self.totalFolds = totalFolds
        self.usr2itemsIndxTrain = usr2itemsIndxTrain
        self.usr2itemsIndxValid = usr2itemsIndxValid
        self.silence = silence # For showing process status
        self.predictTrain = predictTrain
        self.write2File = write2File
        self.DEBUG = DEBUG # For numeric debugging
        self.MAX_TRAIN_NUM = MAX_TRAIN_NUM
        self.NegSampleRun = self.MAX_TRAIN_NUM * 0.9
        self.ITEM_FIELDS_NUM = ITEM_FIELDS_NUM
        self.createFile = False

        # Stats for conv
        self.t = 0
        self.microF1Valid = 0.0  # In [0,1]; bigger <=> higer precision, recall

        # Stats for debugging
        self.avgloss = float('inf')
        self.prevAvgLoss = float('inf')
        self.lossDiff = float('inf')

        # Optimization configs
        self.incrInd = False

    def logFoldInfo(self):
        self.__log__(LogFlags['INFO'] + '(curFold, totalFolds) = ' + str(self.curFold) + ',' + str(self.totalFolds))
        self.__log__(LogFlags['INFO'] + 'usrs in train: ' + str(len(self.usr2itemsIndxTrain)))
        self.__log__(LogFlags['INFO'] + 'usrs in valid: ' + str(len(self.usr2itemsIndxValid)))

    def logStartNegSample(self):
        self.__log__(LogFlags['INFO'] + 'resample usr negative samples')

    def logNegSampleInfo(self, usr2NegativeSamples):
        self.__log__(LogFlags['INFO'] + 'diff' + str(self.lossDiff))
        self.__log__(LogFlags['INFO'] + 'usr2NegativeSamples: ')
        for usrid in usr2NegativeSamples:
            print usrid
            for sample in usr2NegativeSamples[usrid]:
                print ' ', sample

    def logLossStates(self, W, V, avgloss):
        self.__log__(LogFlags['INFO'] + 'at run == ' + str(self.t))
        self.__log__(LogFlags['INFO'] + 'avgloss (only for train -- since this requires samples) == ' + str(avgloss))
        self.__log__(LogFlags['INFO'] + 'loss diff == ' + str(self.lossDiff))
        self.__log__(LogFlags['INFO'] + 'W[:,0] == ' + str(W[:, 0]))
        self.__log__(LogFlags['INFO'] + 'V == ' + str(V))

    def logStartPrediction(self):
        self.__log__(LogFlags['INFO'] + 'start predicting, time == ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        if not self.predictTrain:
            self.__log__(LogFlags['INFO'] + 'will NOT predict on train data')

    def logCollectingStats(self):
        self.__log__(LogFlags['INFO'] + 'Start collecting stats, time == ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    def logStats(self, data):
        name = data['name']
        KPIs = data['KPIs']
        for kpi, val in KPIs.iteritems():
            s = LogFlags['INFO'] + ' '.join([name, 'data', kpi, '==', str(val)])
            self.__log__(s)

    def writeCSVStats(self, data):
        # Format: dataset, latentVarsNum, fold, curRun, spentTime, train/valid, metric, val
        dataset = self.dataset + self.datasetSub if self.datasetSub else self.dataset
        latentVarsNum = self.ITEM_FIELDS_NUM
        fold = self.curFold
        curRun = self.t
        spentTime = int(time() - self.initTime)  # In secs
        splitDataName = data['name']
        baseCols = [dataset, latentVarsNum, fold, curRun,
            spentTime, splitDataName] 
        baseCols = [str(s) for s in baseCols]
        KPIs = data['KPIs']

        outFile = self.RootDir + ''.join([str(latentVarsNum), 'F', dataset])
        self.__log__(LogFlags['INFO'] + 'write to' + outFile)
        flag = 'a' if self.curFold != 0 or self.createFile else 'w+'  # Overwrite at first fold
        self.createFile = True
        with open(outFile, flag) as f:
            for kpi, val in KPIs.iteritems():
                cols = baseCols[:]
                cols += [kpi, str(val)]
                l = ','.join(cols) + '\n'
                f.write(l)

    def logRealPredictedVals(self, data):
        usr2itemsIndx = data['usr2itemsIndx']
        usr2NonzeroCols = data['usr2NonzeroCols']
        u2predictions = data['u2predictions']
        name = data['name']
        self.__log__(LogFlags['INFO'] + ' '.join(['for', name, 'data, print real vals & predicted vals ... ']))
        self.__log__(LogFlags['INFO'] + 'usrid, actual, predicted')
        for usrid in usr2itemsIndx:
            y_nonzeroCols = usr2NonzeroCols[usrid]
            bestCols = u2predictions[usrid]
            print usrid, y_nonzeroCols, bestCols

    def debug(self, msg, val):
        if self.DEBUG:
            self.__log__(LogFlags['INFO'] + ' '.join([msg, str(val)]))

    def __log__(self, s):
        if not self.silence:
            print s

    # States-related funcs
    def nextRun(self):
        self.t += 1
        self.__log__(LogFlags['INFO'] + '### run == ' + str(self.t))
        self.__log__(LogFlags['INFO'] + '### time == ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    def notConv(self):
        # MicroF1 btw [0,1]; bigger => higer precision, higer recall
        return self.microF1Valid < self.MF1Bd and self.t <= self.MAX_TRAIN_NUM

    def shouldNegSample(self):
        # Cond1: many runs and long way to go
        # Cond2: |loss' diff| small (perhaps local min)
        return (self.t >= self.EnoughRuns and self.t == self.NegSampleRun) or \
               (fabs(self.lossDiff) < self.LossDiffBd and self.t % self.NegSampleFreq == 0)

    def shouldRevealStats(self):
        return self.t % self.RevealRun == 0

    def shouldCalLoss(self):
        return self.t % self.CalLossRun == 0

    def updateLossState(self, avgloss):
        self.avgloss = avgloss
        self.lossDiff = self.avgloss - self.prevAvgLoss 
        self.prevAvgLoss = self.avgloss

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
