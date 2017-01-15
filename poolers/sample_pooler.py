import sys
import traceback
sys.path.insert(0, '../')

class sample_pooler:
    # return data as: {
    #     usr_i: [ 
    #         embeded features for item_i0 (list of floats),
    #         ...
    #         embeded features for item_in,
    #     ], ...
    # }
    def pool_all(self, usr2itemsfeatures):
        usr2representation = {}
        for usr in usr2itemsfeatures:
            usr2representation[usr] = self.pool(usr2itemsfeatures[usr])
        return usr2representation

    # implemented as average pooling
    def pool(self, itemsfeatures):
        counts = len(itemsfeatures)
        sum_all = reduce(lambda item_obj1, item_obj2: [x + y for x, y in zip(item_obj2['features'], item_obj2['features'])], itemsfeatures)
        return map(lambda v: v/counts, sum_all)
