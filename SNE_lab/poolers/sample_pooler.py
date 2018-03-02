import numpy as np


class sample_pooler:
    '''Sample pooler implementing avg pooling
    >>> pooler = sample_pooler()
    '''

    # return data as: {
    #     usr_i: [
    #         embeded features for item_i0 (list of floats),
    #         ...
    #         embeded features for item_in,
    #     ], ...
    # }
    def pool_all_inits(self, usr2itemsfeatures):
        usr2representation = {}
        for usr in usr2itemsfeatures:
            usr2representation[usr] = self.pool(usr2itemsfeatures[usr])
        return usr2representation

    # implemented as average pooling
    def pool(self, itemsfeatures):
        counts = len(itemsfeatures)
        sum_all = reduce(lambda item_obj1, item_obj2: [x + y for x, y in zip(item_obj2['features'], item_obj2['features'])], itemsfeatures)
        return map(lambda v: v/counts, sum_all)

    # return usr pooled features (row vector), by averaging
    def pool_all(self, itemsIndx, V):
        # items_representation = [V[itemIndx] for itemIndx in itemsIndx]
        # usr_representation = reduce(lambda item_rep0, item_rep1: [x + y for x, y in zip(item_rep0, item_rep1)], items_representation)
        # return usr_representation
        items_representation = V[itemsIndx, :]
        usr_representation = np.sum(items_representation, axis=0) * (1.0 / len(itemsIndx))
        return usr_representation


if __name__ == '__main__':
    import doctest
    doctest.testmod()
