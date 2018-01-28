import traceback

class Baseloader(object):
    def __init__(self, rating_file, usr2labels_file):
        self.rating_file = rating_file
        self.usr2labels_file = usr2labels_file

    # All rating files have the same format
    # XXX really that tedious ???
    def load(self):
        usr2itemsIndx = {}
        itemsList = []
        f = open(self.rating_file, 'r')
        for line in f:
            try:
                line = line.strip().split(' ')
                usr = int(line[0])
                itemNum = int(line[1])

                # init usr, item (if needed)
                usr2itemsIndx[usr] = usr2itemsIndx[usr] if usr in usr2itemsIndx else []
                itemIndx = len(itemsList)
                if itemNum in itemsList:
                    itemIndx = itemsList.index(itemNum)
                else:
                    itemsList.append(itemNum)
                
                usr2itemsIndx[usr].append(itemIndx)
            except:
                print traceback.format_exc()
                pass
        f.close()

        ind2ItemNum = {k: v for v, k in enumerate(itemsList)}
        return usr2itemsIndx, ind2ItemNum

    ## Return each usr's labels from 'usrs'
    def get_labels(self, usrs):
        usr2labels = {}
        usr2nonZeroCols = {}
        fd = open(self.usr2labels_file, 'r')
        for line in fd:
            try:
                line = line.strip().split(',')
                usr = int(line[0])

                if usr in usrs:
                    # get formaulated labels
                    usr2labels[usr] = [int(e) for i, e in enumerate(line[1:])] 
            except:
                print traceback.format_exc()
                raise
        fd.close()
        return usr2labels

    # Return each usr's nonZeroCols
    def get_nonZeroCols(self):
        usr2labels = {}
        usr2nonZeroCols = {}
        fd = open(self.usr2labels_file, 'r')
        for line in fd:
            try:
                line = line.strip().split(',')
                usr = int(line[0])

                usr2nonZeroCols[usr] = [i for i, e in enumerate(line[1:]) if int(e) != 0] 
            except:
                print traceback.format_exc()
                raise
        fd.close()
        return usr2nonZeroCols

    def gettotalLabelsNum(self):
        with open(self.usr2labels_file, 'r') as f:
            first_line = f.readline()
        return len(first_line.strip().split(',')) - 1

