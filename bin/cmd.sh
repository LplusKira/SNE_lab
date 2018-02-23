#!/bin/bash
source utils.sh
REPORTDIR="../report/"
MAINDIR="../SNE_lab/"
USAGE="[Err] Usage: $0 data|ego|train|valid [file] [avgPrec|microF1|coverage|hammingLoss|RL|oneError]"

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
elif [ "$1" = "train" ] || [ "$1" = "valid" ]; then
  if [ "$2" = "" ] || [ "$3" = "" ]; then
    echo ${USAGE}
    exit 1
  fi
  echo "[info] Inspect $2 trainning results by evaluating $1's data on $3"
  # Only show 1st fold for now
  ## Temp dir to save stats
  dir=`mktemp -d`

  ## Save stats (file)
  trainValid=$1
  raw=$2
  metric=$3
  stats="${dir}/stats"
  awk -F, -v d=${trainValid} -v m=${metric} \
    -v fold=3 -v trainValid=7 -v metric=8 -v run=4 -v val=9 '{\
    if($trainValid == d && $metric == m && $fold == 0) \
      print $run","$val; \
    }' $raw > ${stats}
  gnuplot -e "filename='${stats}'" -e "myTitle='${raw}:${trainValid}:${metric}'" \
    -e "xaxis='run'" plot.sh

  ## Clean stats (file)
  rm -r ${dir}
else
  echo ${USAGE}
fi
