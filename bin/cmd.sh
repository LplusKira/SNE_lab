#!/bin/bash
source utils.sh
REPORTDIR="../report/"
MAINDIR="../SNE_lab/"

## Assert the execution place
assertCurDir $0

if [ "$1" = "ego" ]; then
  echo "[info] Will iterate over ego-net's 10 dataSets"
  DATADIR="../data/ego-net/"
  LEARNING_RATE="0.005"
  FoldNum="5"
  MAX_TRAIN_NUM="1000"
  cd ${MAINDIR} && {
    for tag in 348 1912 0 107 1684 3980 414 686 698 3437; do
      for featNum in 100 200 300 400; do
        # Would 'roughly' run four processes (by featNum)
        if [ "${featNum}" -ne "400" ]; then
          NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=${featNum} MAX_TRAIN_NUM=${MAX_TRAIN_NUM} LEARNING_RATE=${LEARNING_RATE} MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 ${FoldNum} ego-net rating_file="../data/ego-net/${tag}.edges.u2u" usr2labels_file="../data/ego-net/${tag}.circles.u2f.filtered" sub=${tag} > "${REPORTDIR}${featNum}Fego-net${tag}_out" &
        else
          NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=${featNum} MAX_TRAIN_NUM=${MAX_TRAIN_NUM} LEARNING_RATE=${LEARNING_RATE} MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py 0 ${FoldNum} ego-net rating_file="../data/ego-net/${tag}.edges.u2u" usr2labels_file="../data/ego-net/${tag}.circles.u2f.filtered" sub=${tag} > "${REPORTDIR}${featNum}Fego-net${tag}_out"
        fi
      done
    done
    cd -;
  }
elif [ "$1" = "data" ]; then
  echo "[info] Render all dataset"
  bash ml-100k.sh
  bash ml-1m.sh
  bash ego-net.sh
  bash youtube.sh
else
  echo "[Err] Usage: $0 data|ego"
fi
