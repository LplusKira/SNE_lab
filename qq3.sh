#!/bin/bash
#time NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=300 MAX_TRAIN_NUM=10000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 python run.py data/usrItemRating_1st10K > report/300F 2> report/time300F
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=300 MAX_TRAIN_NUM=10000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 python run.py data/1m/ratings.dat.filtered > report/300F1m 2> report/time300F1m

