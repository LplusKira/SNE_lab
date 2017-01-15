import dataloaders.movielens100k as dataloader_movielens100k
import embedders.sample_embedder as sample_embedder
import poolers.sample_pooler as sample_pooler
from utils import get_labels, load_data

import numpy as np

def main():
    # load data as {
    #     usr_i(int): [ item_i0(int), item_i1(int), ... ]
    #     ... 
    # }
    dataloader = dataloader_movielens100k.dataloader_movielens100k()
    usr2items = dataloader.load('data/u.template.BOI')

    # embed data as: {
    #     usr_i: [{ 
    #             item_id: item_i0,
    #             features: embeded features for item_i0 (list of floats),
    #         },{
    #         ...}, {
    #             item_id: item_in,
    #             features: embeded features for item_in,
    #         }
    #     ], ...
    # }
    embedder = sample_embedder.sample_embedder()
    usr2itemsfeatures = embedder.embed_all(usr2items)

    # pool data as: {
    #     usr_i: usr_i's representation list,
    #     ....
    # }
    pooler = sample_pooler.sample_pooler()
    usr2representation = pooler.pool_all(usr2itemsfeatures)

    # acquire usr labels as: {
    #     usr_i: demographic labels (list of list(could be empty one)),
    #     ...
    # }
    usr2labels = get_labels('data/u.template.labels')

    # assemble X_train as: [
    #     usr_i's representation,
    #     ...
    # ]; similar for y_train -- they have the same order w.r.t usr
    X_train, y_train = load_data(usr2labels, usr2representation)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print 'x', X_train, X_train.shape
    print 'y', y_train, y_train.shape

if __name__ == '__main__':
    main()
