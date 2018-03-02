from BaseValidator import BaseValidator


class ENValidator(BaseValidator):
    def __init__(self, dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, predictTrain=False, write2File=True):
        super(ENValidator, self).__init__(dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, predictTrain=predictTrain, write2File=write2File)

        # Customized params
        self.RevealRun = min(500, self.MAX_TRAIN_NUM/2)
        self.CalLossRun = min(500, self.RevealRun)
