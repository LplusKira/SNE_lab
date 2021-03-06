from Baseloader import Baseloader
from os import environ


class ML100KLoader(Baseloader):

    def __init__(self, rating_file='../data/ml-100k/u.data.filtered', usr2labels_file='../data/ml-100k/u.user.one.filtered', sub=None):
        super(ML100KLoader, self).__init__(rating_file, usr2labels_file, sub)
        self.attr_bds = [[0, 3], [4, 5], [6, 26]]

        # Training config
        self.NEG_SAMPLE_NUM = int(environ.get('NEG_SAMPLE_NUM', 1))
        self.ITEM_FIELDS_NUM = int(environ.get('ITEM_FIELDS_NUM', 100))
        self.MAX_TRAIN_NUM = int(environ.get('MAX_TRAIN_NUM', 1000))
        self.LEARNING_RATE = float(environ.get('LEARNING_RATE', 1E-4))  # Update by (1 OR 1/MOMENTUM) * LEARNING_RATE * gradient
        self.MOMENTUM = float(environ.get('MOMENTUM', 1.0))
        self.LAMBDA = float(environ.get('LAMBDA', 1E-3))

    # The one-hot encoding boundary cols for each attribute
    def getBds(self):
        return self.attr_bds

    def getRLPairsCnt(self):
        total = self.gettotalLabelsNum()
        return (total - len(self.attr_bds)) * len(self.attr_bds)
