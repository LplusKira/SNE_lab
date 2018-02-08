from BaseValidator import BaseValidator
class ML1MValidator(BaseValidator):
    def __init__(self, dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, silence=False, predictTrain=False, write2File=True):
        super(ML1MValidator, self).__init__(dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, silence=False, predictTrain=False, write2File=True)

        # Customized params
        setRuns(RevealRun=500, CalLossRun=500, cb=setCalLossRun)
        self.microF1Valid = 0.5
        self.NegSampleFreq = 5

    def setRuns(self, RevealRun, CalLossRun, cb):
        # Should at least reveal twice if running all runs
        lowerBd = self.MAX_TRAIN_NUM/2
        self.RevealRun = min(RevealRun, lowerBd)
        self.CalLossRun = cb(CalLossRun)

    def setCalLossRun(self, val):
        # Should at least as frequent as revealing run
        lowerBd = self.RevealRun
        return min(val, self.RevealRun)
