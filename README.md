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
  ITEM_FIELDS_NUM=4 MAX_TRAIN_NUM=100 LEARNING_RATE=0.0002 MOMENTUM=2.0 LAMBDA=1 python run.py data/yourTrainFile
  ```

# structure:
  refer to 'Your Cart tells You: Inferring Demographic Attributes from Purchase Data'

# data format:
  see data formulated after 'bash cmd.sh' under data/; each line is:
  ```
  usrid	itemid	rating	somethingNotImportant
  ```

# procedure of adding modules <-- if you have time:
  ```
  0. add the module and its test in `test.py`
  ```
  ```
  1. add the required lib in `requirements.txt`
  ```

# ref:
  0. the dataset is from http://files.grouplens.org/datasets/movielens/ml-100k.zip

# collector
