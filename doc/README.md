# Reproduce Paper Results:
## ML-100K (__run over__ ITEM_FIELDS_NUM 100, 200, 300, 400):
```
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=400 MAX_TRAIN_NUM=8000 \
LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 \
nohup python -u sne_lab.py 0 10 ml-1m > ../report/400Fout &
```
## ML-1M (__run over__ ITEM_FIELDS_NUM 100, 200, 300, 400):
```
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=400 MAX_TRAIN_NUM=1000 \
LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 \
nohup python -u sne_lab.py 0 10 ml-1m > ../report/400Fml1mout &
```
## Ego-net (Facebook):
```
# Modify FoldNum & MAX_TRAIN_NUM to "10" & "2000"
bash cmd ego
``` 
## Youtube (__run over__ ITEM_FIELDS_NUM 100, 200, 300, 400):
```
NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=400 MAX_TRAIN_NUM=1500 \
LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 \
nohup python -u sne_lab.py 0 10 youtube > ../report/400Fout &
```
