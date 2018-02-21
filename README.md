# Install:
  ```
  pip install -r requirements.txt
  ```
  ```
  XXX bash cmd.sh
  ```
# Run:
  ```
  # MovieLens100K, 5-folds cv
  NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=8000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 5 ml-100k

  # MovieLens1M, 5-folds cv
  NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=1000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 5 ml-1m

  # Youtube, 5-folds cv
  NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=2000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 5 youtube

  # Ego-net (Facebook)'s network#348, 5-folds cv
  NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=1000 LEARNING_RATE=0.005 MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 5 ego-net rating_file="../data/ego-net/348.edges.u2u" usr2labels_file="../data/ego-net/348.circles.u2f.filtered" sub=348
  ```
# Examine:
  ```
  bash genStats.sh ml-100k|ml-1m|youtube|ego-net348 foldNum
  ```
  ```
  XXX ## gnuplot is required
  run cmd.sh ## e.g. ./cmd.sh train 1err
  ```

# Structure:
  Refer to [Your Cart tells You: Inferring Demographic Attributes from Purchase Data](https://github.com/LplusKira/SNE_lab/blob/master/doc/WSDM2016_wang.pdf)

# Data format:
  see data formulated after 'bash cmd.sh' under data/; each line is:
  ```
  usrid	itemid	rating	somethingNotImportant
  ```

# How to add modules:
  ```
  0. add the module and its test in `test.py`
  ```
  ```
  1. add the required lib in `requirements.txt`
  ```

# Ref:
  0. [Dataset](http://files.grouplens.org/datasets/movielens/ml-100k.zip) for ml-100k
  1. [Dataset](http://files.grouplens.org/datasets/movielens/ml-1m.zip) for ml-1m
  2. [Dataset](http://snap.stanford.edu/data/facebook.tar.gz) for ego-net
  3. [Graph Data](http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz) and [Community Data](http://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz) for youtube

# TODO
# refactoring:
## scipy/numpy usage
## matplot (instead of gnuplot)
## (after this, merge 1M yelp to master)
# Readme: data flow
# test
# liscence
```
  wtf --> ??? --> XXX
```
# linting?
# vim python folding
# W, V should be initiated as float
# better test

# warn:
  you should exactly see no warning/err
  (if you do, it's plaussibly the weights somehow overlow)

