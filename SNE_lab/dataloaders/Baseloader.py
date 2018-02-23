from sys import path
path.append('../')  # To get config from previous layer
from config import LogFlags

class Baseloader(object):
    def __init__(self, rating_file, usr2labels_file, sub=None, silence=False):
        self.rating_file = rating_file
        self.usr2labels_file = usr2labels_file
        self.sub = sub  # Subtitle of (rating_file, usr2labels_file) comb
        self.silence = silence
        if not silence:
            print self.__str__()

        # For training (default template)
        self.NEG_SAMPLE_NUM  = 10
        self.ITEM_FIELDS_NUM = 2
        self.MAX_TRAIN_NUM   = 10
        self.LEARNING_RATE   = 1E-4  # Update by (1 OR 1/MOMENTUM) * LEARNING_RATE * gradient
        self.MOMENTUM        = 2.0
        self.LAMBDA          = 1E-3

    # Get description of loaded data
    def getDataSub(self):
        return self.sub

    # All rating files have the same format
    #  ind2itemNum = {
    #    0: 5566,
    #    1: 87,
    #    2: 2266,
    #  }
    #  usr2itemsIndx = {
    #    0: [0,1],
    #    1: [1,2],
    #  }
    def load(self):
        usr2itemsIndx = {}
        itemsList = []
        f = open(self.rating_file, 'r')
        for line in f:
            # No try-except here: terminate for corruption/kill
            line = line.strip().split(',')
            usr = int(line[0])
            itemNum = int(line[1])

            # Init this usr's bag if needed
            usr2itemsIndx[usr] = usr2itemsIndx[usr] if usr in usr2itemsIndx else []

            # Append new item; index it
            itemIndx = len(itemsList)
            if itemNum in itemsList:
                itemIndx = itemsList.index(itemNum)
            else:
                itemsList.append(itemNum)
            
            # Append item's 'indx' to usr's bag
            usr2itemsIndx[usr].append(itemIndx)
        f.close()

        ind2ItemNum = {k: v for v, k in enumerate(itemsList)}
        self.__log__(LogFlags['INFO'] + 'usr2itemsIndx loaded')
        return usr2itemsIndx, ind2ItemNum

    ## Return each usr's labels/nonzeroCols from 'usrs'
    # sample:
    # usr2labels = {
    #   0: [0,0,1, 1,0],
    #   1: [1,0,0, 0,1],
    # }
    # usr2NonzeroCols = {
    #   0: [2, 3],
    #   1: [0, 4],
    # }
    def get_labels(self, usrs):
        usr2labels = {}
        usr2nonZeroCols = {}
        fd = open(self.usr2labels_file, 'r')
        for line in fd:
            # No try-except here: terminate for corruption/kill
            line = line.strip().split(',')
            usr = int(line[0])
            if usr in usrs:
                # Get one-hot attributes
                usr2labels[usr] = [int(e) for i, e in enumerate(line[1:])] 
                usr2nonZeroCols[usr] = [i for i, e in enumerate(line[1:]) if int(e) != 0] 
        fd.close()
        self.__log__(LogFlags['INFO'] + 'usr2labels and usr2NonzeroCols loaded')
        return usr2labels, usr2nonZeroCols

    def gettotalLabelsNum(self):
        with open(self.usr2labels_file, 'r') as f:
            first_line = f.readline()
        return len(first_line.strip().split(',')) - 1

    def __repr__(self):
        return "{}(rating_file='{}', usr2labels_file='{}', silence={})".format(
            self.__class__.__name__, 
            self.rating_file, 
            self.usr2labels_file,
            self.silence,
        )

    def __str__(self):
        s = LogFlags['INFO'] + "Load BOI from '{}'\n" + \
            LogFlags['INFO'] + "Load one-hot-encoded attributes from '{}'"
        return s.format(self.rating_file, self.usr2labels_file)
    
    def __log__(self, s):
        if not self.silence:
            print s

    def getTrainingConf(self):
        return self.NEG_SAMPLE_NUM, \
            self.ITEM_FIELDS_NUM, \
            self.MAX_TRAIN_NUM, \
            self.LEARNING_RATE, \
            self.MOMENTUM, \
            self.LAMBDA
