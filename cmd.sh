#!/bin/bash
data="/Users/lpluskira/lab/SNE_lab/report/200Feat"  # where you save the log
#data="/Users/lpluskira/lab/SNE_lab/report/400F"  # where you save the log
dir=`mktemp -d`                                # tmp dir for saving plot data

if [ "$1" == "train" ]; then
  if [ "$2" != "" ]; then
    if [ "$2" == "f1" ]; then
      awk -F'train data microF1 == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
    elif [ "$2" == "1err" ]; then
      awk -F'train data oneError == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
    elif [ "$2" == "rl" ]; then
      awk -F'train data RL == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
    else
      echo "[usage]: bash $0 train/valid f1|1err|rl"
      exit 1
    fi
    
    #echo "${dir}/stats"
    gnuplot -e "filename='${dir}/stats'" plot.sh
    rm -rf "${dir}/stats"
  else
    echo "[usage]: bash $0 train/valid f1|1err|rl"
  fi
elif [ "$1" == "valid" ]; then
  if [ "$2" != "" ]; then
    if [ "$2" == "f1" ]; then
      awk -F'valid data microF1 == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
    elif [ "$2" == "1err" ]; then
      awk -F'valid data oneError == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
    elif [ "$2" == "rl" ]; then
      awk -F'valid data RL == ' -v a=0 '{if($2 != "") { a += 1; print a","$2; }}' ${data} > "${dir}/stats"
    else
      echo "[usage]: bash $0 train/valid f1|1err|rl"
      exit 1
    fi
    
    #echo "${dir}/stats"
    gnuplot -e "filename='${dir}/stats'" plot.sh
    rm -rf "${dir}/stats"
  else
    echo "[usage]: bash $0 train/valid f1|1err|rl"
  fi
else
  echo "[usage]: bash $0 train/valid f1|1err|rl"
fi

