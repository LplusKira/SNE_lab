from Baseloader import Baseloader
from os import environ


class ENLoader(Baseloader):

    def __init__(self, rating_file='../data/ego-net/3980.edges.u2u', usr2labels_file='../data/ego-net/3980.circles.u2f.filtered', sub='3980', silence=False):
        super(ENLoader, self).__init__(rating_file, usr2labels_file, sub, silence)
        self.attr_bds = self.attrBdsInit()

        # Training config
        self.NEG_SAMPLE_NUM = int(environ.get('NEG_SAMPLE_NUM', 1))
        self.ITEM_FIELDS_NUM = int(environ.get('ITEM_FIELDS_NUM', 100))
        self.MAX_TRAIN_NUM = int(environ.get('MAX_TRAIN_NUM', 100))
        self.LEARNING_RATE = float(environ.get('LEARNING_RATE', 1E-4))  # Update by (1 OR 1/MOMENTUM) * LEARNING_RATE * gradient
        self.MOMENTUM = float(environ.get('MOMENTUM', 2.0))
        self.LAMBDA = float(environ.get('LAMBDA', 1E-3))

    def attrBdsInit(self):
        attr_bds = []
        cols = range(self.gettotalLabelsNum())
        for i in range(len(cols) / 2):
            attr_bds.append([2 * i, 2 * i + 1])
        return attr_bds

    # The one-hot encoding boundary cols for each attribute
    # - for ENLoader, should be all binary
    def getBds(self):
        return self.attr_bds

    def getRLPairsCnt(self):
        total = self.gettotalLabelsNum()
        return (total - len(self.attr_bds)) * len(self.attr_bds)
