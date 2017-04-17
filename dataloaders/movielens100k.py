import sys
import traceback
from bisect import bisect_left
from config import AGES_BOUNDERIES
sys.path.insert(0, '../')

class dataloader_movielens100k:
    def __init__(self):
        self.ageindxLabelsDict = {
            0: [1,0,0,0],
            1: [0,1,0,0],
            2: [0,0,1,0],
            3: [0,0,0,1],
        }

        self.genderLabelDict = {
          'M': [1,0],
          'F': [0,1],
        }

        self.occupationLabelsDict = {}
        for i in range(21):
            tmp = [0] * 21
            tmp[i] = 1
            self.occupationLabelsDict[i+1] = tmp

    def load(self, file_path):
        usr2itemsIndx = {}
        itemsList = []
        f = open(file_path, 'r')
        for line in f:
            try:
                line = line.strip().split('\t')
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

    ## get each, in usrs, usr's labels 
    def get_labels(self, usr2labels_file, usrs):
        usr2labels = {}
        usr2nonZeroCols = {}
        fd = open(usr2labels_file, 'r')
        for line in fd:
            try:
                line = line.strip().split('|')
                usr = int(line[0])

                if usr in usrs:
                    # get labels for age
                    age = int(line[1])
                    ageInd = bisect_left(AGES_BOUNDERIES, age)
                    ageLabels = self.ageindxLabelsDict[ageInd]
                    
                    # get labels for gender
                    gender = line[2]
                    genderLabels = self.genderLabelDict[gender] 

                    # get labels for occupation
                    occupation = int(line[3])
                    occupationLabels = self.occupationLabelsDict[occupation]

                    usr2labels[usr] = ageLabels + genderLabels + occupationLabels
            except:
                print traceback.format_exc()
                raise
        fd.close()
        return usr2labels

    # not done with get_labels to avoid 'too many values to unpack'
    def get_nonZeroCols(self, usr2labels_file):
        usr2labels = {}
        usr2nonZeroCols = {}
        fd = open(usr2labels_file, 'r')
        for line in fd:
            try:
                line = line.strip().split('|')
                usr = int(line[0])

                # get labels for usr
                age = int(line[1])
                ageInd = bisect_left(AGES_BOUNDERIES, age)
                
                # get labels for gender
                gender = line[2]
                genderInd = 4 if gender == 'M' else 5

                # get labels for occupation
                occupation = int(line[3])
                occupationInd = 5 + occupation

                usr2nonZeroCols[usr] = [ageInd, genderInd, occupationInd]
            except:
                print traceback.format_exc()
                raise
        fd.close()
        return usr2nonZeroCols

    def gettotalLabelsNum(self):
        return 4 + 2 + 21

