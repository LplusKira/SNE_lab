#!/bin/bash
#time NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=10000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 python run.py data/usrItemRating_1st10K > report/100F1m 2> report/time100F1m
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=10000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 python run.py data/1m/ratings.dat.filtered > report/100F1m 2> report/time100F1m

