#!/bin/bash
cwd=`pwd`
DATADIR="../data/ego-net/"
REPORTDIR="../report/"
MAINDIR="../SNE_lab/"
LEARNING_RATE="0.01"

cd ${MAINDIR}
#for tag in "3980" "414" "686" "698"; do
#for tag in "0" "107" "1684"; do
for tag in "3437" "348" "1912"; do
  for featNum in 100 200 300 400; do
    if [ "${featNum}" -ne "400" ]; then
      NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=${featNum} MAX_TRAIN_NUM=5000 LEARNING_RATE=${LEARNING_RATE} MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 10 ego-net rating_file="../data/ego-net/${tag}.edges.u2u" usr2labels_file="../data/ego-net/${tag}.circles.u2f.filtered" sub=${tag} > "${REPORTDIR}${featNum}Fego-net${tag}_out" &
    else
      NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=${featNum} MAX_TRAIN_NUM=5000 LEARNING_RATE=${LEARNING_RATE} MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 10 ego-net rating_file="../data/ego-net/${tag}.edges.u2u" usr2labels_file="../data/ego-net/${tag}.circles.u2f.filtered" sub=${tag} > "${REPORTDIR}${featNum}Fego-net${tag}_out"
    fi
  done
done
cd ${cwd}
