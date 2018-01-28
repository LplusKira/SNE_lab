#!/bin/bash
#data="/Users/lpluskira/lab/SNE_lab/report/100F"  # where you save the log
#data="/Users/lpluskira/lab/SNE_lab/report/200F"  # where you save the log
#data="/Users/lpluskira/lab/SNE_lab/report/300F"  # where you save the log
#data="/Users/lpluskira/lab/SNE_lab/report/400F"  # where you save the log
#dir=`mktemp -d`                                # tmp dir for saving plot data

#if [ "$1" == "train" ]; then
#  if [ "$2" != "" ]; then
#    if [ "$2" == "f1" ]; then
#      awk -F'train data microF1 == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    elif [ "$2" == "1err" ]; then
#      awk -F'train data oneError == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    elif [ "$2" == "rl" ]; then
#      awk -F'train data RL == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    elif [ "$2" == "cv" ]; then
#      awk -F'train data coverage == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    elif [ "$2" == "prec" ]; then
#      awk -F'train data avgPrec == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    else
#      echo "[usage]: bash $0 train/valid f1|1err|rl|cv|prec"
#      exit 1
#    fi
#    
#    #echo "${dir}/stats"
#    gnuplot -e "filename='${dir}/stats'" plot.sh
#    rm -rf "${dir}/stats"
#  else
#    echo "[usage]: bash $0 train/valid f1|1err|rl"
#  fi
#elif [ "$1" == "valid" ]; then
#  if [ "$2" != "" ]; then
#    if [ "$2" == "f1" ]; then
#      awk -F'valid data microF1 == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    elif [ "$2" == "1err" ]; then
#      awk -F'valid data oneError == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    elif [ "$2" == "rl" ]; then
#      awk -F'valid data RL == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    elif [ "$2" == "cv" ]; then
#      awk -F'valid data coverage == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    elif [ "$2" == "prec" ]; then
#      awk -F'valid data avgPrec == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
#    else
#      echo "[usage]: bash $0 train/valid f1|1err|rl"
#      exit 1
#    fi
#    
#    #echo "${dir}/stats"
#    gnuplot -e "filename='${dir}/stats'" plot.sh
#    rm -rf "${dir}/stats"
#  else
#    echo "[usage]: bash $0 train/valid f1|1err|rl|cv|prec"
#  fi
#else
#  echo "[usage]: bash $0 train/valid f1|1err|rl|cv"
#fi

cwd=`pwd`
DATADIR="../data/ego-net/"
REPORTDIR="../report/"
MAINDIR="../SNE_lab/"
 
cd ${MAINDIR}
#for tag in "3980" "414" "686" "698"; do
#for tag in "1912" "3437" "348"; do
for tag in "0" "107" "1684"; do
  for featNum in 100 200 300 400; do
    for foldNum in {1..10}
    do
      NEG_SAMPLE_NUM=1 ITEM_FIELDS_NUM=${featNum} MAX_TRAIN_NUM=1000 LEARNING_RATE=0.001 MOMENTUM=1.0 LAMBDA=0.001 nohup python -u sne_lab.py ${foldNum} "${DATADIR}${tag}.edges.u2u" "${DATADIR}${tag}.circles.u2f.filtered" > "${REPORTDIR}${featNum}Fego-net_${foldNum}" &
      #echo $tag $featNum $foldNum
    done
  done
done
cd ${cwd}
