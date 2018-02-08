from BaseValidator import BaseValidator
class YTValidator(BaseValidator):

    def __init__(self, dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, silence=False, predictTrain=False, write2File=True):
        super(YTValidator, self).__init__(dataset, datasetSub, curFold, totalFolds, usr2itemsIndxTrain, usr2itemsIndxValid, MAX_TRAIN_NUM, ITEM_FIELDS_NUM, silence=False, predictTrain=False, write2File=True)

        # Customized params
        self.RevealRun = 1000
        self.CalLossRun = 2000

