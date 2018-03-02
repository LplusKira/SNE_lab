from BaseValidator import BaseValidator

class YTValidator(BaseValidator):

    def __init__(self, dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, predictTrain=False, write2File=True):
        super(YTValidator, self).__init__(dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, predictTrain=predictTrain, write2File=write2File)

        # Customized params
        self.RevealRun = 100
        self.CalLossRun = 200
