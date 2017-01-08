# how to run:
  ```
  pip install -r requirements.txt
  ```
  ```
  python `run.py`
  ```

# procedure of adding modules:
  ```
  0. add the module and its test in `test.py`
  ```
  ```
  1. add the required lib in `requirements.txt`
  ```

# structure:
  0. for a dataset X (its  corresponding dataloader dataloader_X)
       load dataset X to some specific oracles
  1. each oracle learns how to 'answere/predict'
  2. load initial train data D0 to mf's model
  3. TODO: the proactive learning's part follows ...

# data format:
  see u.template.BOI(BOI: bag of items) under data/; each line is:
  ```
  someUsrid_i|itemid_i0|itemid_i1|...|itemid_in
  ```

# remark:
  * for any modules written, add test to test/test.py and required test data under test/stubs/
  * dont push data to this proj (test data excluded)
  * since no ci now, just ensure that the test is ok before you push

# ref:
  0. the dataset is from http://files.grouplens.org/datasets/movielens/ml-100k.zip
