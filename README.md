# Install:
```
pip install -r requirements.txt
cd bin/; bash cmd.sh data; cd -;
```

# Run:
- MovieLens100K, 5-fold cv
```
cd SNE_lab/; \
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=8000 \
LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 \
nohup python -u sne_lab.py 0 5 ml-100k; \
cd -;
```

- MovieLens1M, 5-fold cv
```
cd SNE_lab/; \
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=1000 \
LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 \
nohup python -u sne_lab.py 0 5 ml-1m; \
cd -;
```

- Youtube, 5-fold cv
```
cd SNE_lab/; \
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=2000 \
LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 \
nohup python -u sne_lab.py 0 5 youtube; \
cd -;
```

- Ego-net (Facebook)'s network#348, 5-fold cv
```
cd SNE_lab/; \
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=100 MAX_TRAIN_NUM=1000 \
LEARNING_RATE=0.005 MOMENTUM=1.0 LAMBDA=0.001 \
nohup python -u sne_lab.py 0 5 ego-net rating_file="../data/ego-net/348.edges.u2u" usr2labels_file="../data/ego-net/348.circles.u2f.filtered" sub=348; \
cd -
```

# Examine:
- Render report & raw stats in .csv
```
cd bin/; \
bash genStats.sh ml-100k|ml-1m|youtube|ego-net348 foldNum; \
cd -;
```
- Optional: gnuplot for quick visualization from rawFile
```
cd bin/; \
bash cmd.sh train|valid rawFile [avgPrec|microF1|coverage|hammingLoss|RL|oneError] [featNum]; \
cd -
```
[Check definition of rawFile](https://github.com/LplusKira/SNE_lab#inout-format)

# Structure:
- For model, please refer to [Your Cart tells You: Inferring Demographic Attributes from Purchase Data](https://github.com/LplusKira/SNE_lab/blob/master/doc/WSDM2016_wang.pdf)
- For codes' architecture
```
  dataloader     loads data and keeps respective dependencies
  |
  V
  statevalidator handles the process of training (data-dependent)
  |
  V
  updator        is stateless handling parameters' updates

  pooler         is stateless handling pooling part
```

# In/Out format:
Two input files required for sne_lab.py:
- rating_file
- usr2labels_file

Rules:
- rating_file and usr2labels_file should be in CSV with comma delimiter
- each line in rating_file is 'usr' 'item' 'rating'
- each line in usr2labels_file is 'usr' concatenated by one-hot encoded attributes

The corresponding shell scripts in bin/ handles the transformation of raw data to the designated formatted data. For example, ego-net.sh handles the generation of the formatted files for ego-net's datasets.

Will record statistics (rawFile) to report/ and log current status to stdout
'rawFile' format is depicted in statevalidator

# Ref:
0. [Dataset](http://files.grouplens.org/datasets/movielens/ml-100k.zip) for ml-100k
1. [Dataset](http://files.grouplens.org/datasets/movielens/ml-1m.zip) for ml-1m
2. [Dataset](http://snap.stanford.edu/data/facebook.tar.gz) for ego-net
3. [Graph Data](http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz) and [Community Data](http://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz) for youtube

# TODO:
- 'XXX' in files
- Refactoring (matplot, save/load model, exception handling in shell scripts)
- Linting (by flake8)
- Better test
- 'Bot' notification for remanining run time

# Moral:
- Use built-in module logging
- Use 'legit' names (e.g. pseudoSigmoid VS sigmoid) 
- 'Drawbacks' of using environ vars to pass arguments (e.g. when examine process from sys command)
- 'Never' push data/report/something huge
- Use linting as early as possible

# TL;DR:
- Try this simple one first: Will generate __10Fml-100k__ under report/
```
pip install -r requirements.txt
cd bin/; bash cmd.sh data; cd -;
cd SNE_lab/; \
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=10 MAX_TRAIN_NUM=10 \
LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 \
python sne_lab.py 0 2 ml-100k; \
cd -;
```
