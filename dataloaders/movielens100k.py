import sys
import traceback
sys.path.insert(0, '../')

class dataloader_movielens100k:
    def load(self, file_path):
        usr2items = {}
        f = open(file_path, 'r')
        for line in f:
            try:
                line = line.strip().split('|')
                usr = int(line[0])
                usr2items[usr] = usr2items[usr] if usr in usr2items else []
                usr2items[usr] += map(lambda val: int(val), line[1:len(line)])
            except:
                print traceback.format_exc()
                pass
        f.close()
        return usr2items
