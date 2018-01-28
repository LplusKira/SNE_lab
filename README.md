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
  time NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=10000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 python SNE_lab/sne_lab.py data/yourTrainFile > report/qq 2> report/timeqq
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

# TODO
# refactoring:
## architect
## parser
## weird codes suchAs totally hard-coded 'for loops'
## scipy/numpy usage
## matplot (instead of gnuplot)
## (after this, merge 1M yelp to master)
# Readme: data flow
# Do we 'really' need bisect.py?
# liscence
這個dataset不大 裡面有10個ego-network 每個network裡面有user和user的edge作為我們的feature 要預測user屬於哪些social cycle (也可能都不在)
# Architecture
```
  wtf --> ??? --> XXX
```
# linting?
# vim python folding
