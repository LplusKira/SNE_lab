# warn:
  you should exactly see no warning/err
  (if you do, it's plaussibly the weights somehow overlow)

# how to run:
  ```
  pip install -r requirements.txt
  ```
  ```
  bash cmd.sh
  ```
  ```
  ## split your train file
  ```
  ```
  NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=10000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 > report/100Fego-net_0 &
  ```
  ```
  ## gnuplot is required
  run cmd.sh ## e.g. ./cmd.sh train 1err
  ```

# structure:
  refer to 'Your Cart tells You: Inferring Demographic Attributes from Purchase Data'

# data format:
  see data formulated after 'bash cmd.sh' under data/; each line is:
  ```
  usrid	itemid	rating	somethingNotImportant
  ```

# procedure of adding modules <-- no matter what:
  ```
  0. add the module and its test in `test.py`
  ```
  ```
  1. add the required lib in `requirements.txt`
  ```

# ref:
  0. the dataset is from http://files.grouplens.org/datasets/movielens/ml-100k.zip

NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=25 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 python -u sne_lab.py 0 5 ego-net ../data/ego-net/3980.edges.u2u ../data/ego-net/3980.circles.u2f.filtered 3980
# TODO
# refactoring:
## scipy/numpy usage
## matplot (instead of gnuplot)
## (after this, merge 1M yelp to master)
# Readme: data flow
# test
# liscence
# Architecture
```
  wtf --> ??? --> XXX
```
# linting?
# vim python folding
