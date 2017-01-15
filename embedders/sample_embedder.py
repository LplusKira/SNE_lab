import sys
import traceback
sys.path.insert(0, '../')

class sample_embedder:
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
    def embed_all(self, usr2items):
        usr2itemsfeatures = {}
        for usr in usr2items:
            usr2itemsfeatures[usr] = map(lambda item_id: self.embed(item_id), usr2items[usr])
        return usr2itemsfeatures

    def embed(self, item_id):
        return {
            'item_id': item_id,
            'features': [0.0,1.0,2.0],
        }    
