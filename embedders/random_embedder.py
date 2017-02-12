import sys
import traceback
import random
sys.path.insert(0, '../')

class random_embedder:
    # return data as: {
    #     usr_i: [{ 
    #             item_id: item_i0,
    #             features: embeded features for item_i0 (list),
    #         },{
    #         ...}, {
    #             item_id: item_in,
    #             features: embeded features for item_in (list),
    #         }
    #     ], ...
    # }
    def __init__(self):
        cache_init_id2features = {}

    def embed_all(self, usr2items):
        usr2itemsfeatures = {}
        for usr in usr2items:
            usr2itemsfeatures[usr] = map(lambda item_id: self.embed(item_id), usr2items[usr])
        return usr2itemsfeatures

    def embed(self, item_id):
        # use the same init features for the same itemId
        features = cache_init_id2features[item_id] if item_id in cache_init_id2features else [random.random() for i in range(3)]
        cache_init_id2features[item_id] = features
        return {
            'item_id': item_id,
            'features': features,
        }    
