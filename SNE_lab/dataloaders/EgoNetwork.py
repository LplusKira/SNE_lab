from Baseloader import Baseloader

class ENLoader(Baseloader):

    def __init__(self, rating_file, usr2labels_file):
        super(ENLoader, self).__init__(rating_file, usr2labels_file)
        self.attr_bds = self.attrBdsInit()

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
        return  (total - len(self.attr_bds)) * len(self.attr_bds)

