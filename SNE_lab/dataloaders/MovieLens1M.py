from Baseloader import Baseloader
class ML1MLoader(Baseloader):

    def __init__(self, rating_file, usr2labels_file):
        super(ML1MLoader, self).__init__(rating_file, usr2labels_file)
        self.attr_bds = [[0, 6], [7, 8], [9, 29]]

    # The one-hot encoding boundary cols for each attribute
    def getBds(self):
        return self.attr_bds
        
    def getRLPairsCnt(self):
        total = self.gettotalLabelsNum()
        return  (total - len(self.attr_bds)) * len(self.attr_bds)

