#!/bin/bash
source utils.sh
DataSet="$1"
FoldNum="$2"
DataDir="../report/"
RawFE=".raw"
StatsFE=".report"

function genRaw {
  outFile="${DataSet}${FoldNum}${RawFE}.csv"
  rm -f ${outFile}
  for fNum in "100F" "200F" "300F" "400F"; do
    awk -F',' '{k2v[$1","$2","$3","$7","$8] = $NF;} END {for(k in k2v) print k","k2v[k]}' ${fNum}${DataSet} | sort -g -k1,1 -k2,2 -k3,3 -k4,4 -k5,5 >> ${outFile}
  done
}

function genStats {
  outFile="${DataSet}${FoldNum}${StatsFE}.csv"
  rm -f ${outFile}
  for fNum in "100F" "200F" "300F" "400F"; do
    awk -F',' '{k2v[$1","$2","$3","$7","$8] = $NF;} END {for(k in k2v) print k","k2v[k]}' ${fNum}${DataSet} | \
      awk -F',' '{N[$1","$2","$4","$5] += 1; sum[$1","$2","$4","$5] += $NF; sumSqr[$1","$2","$4","$5] += $NF * $NF;} END {for(k in sum) print k","sum[k]","sumSqr[k]","sum[k]/N[k]","((sumSqr[k]/N[k]) - (sum[k]/N[k]) ^ 2) ^ 0.5;}' | \
    sort -g -k1,1 -k2,2 -k3,3 -k4,4 >> ${outFile}
  done
}

## Assert DataSet and FoldNum
if [ "${DataSet}" = "" ] || [ "${FoldNum}" = "" ]; then
  echo "[info] (Usage) $0 dataset foldNum"
  exit 1
fi

## Assert the execution place
assertCurDir $0

## Generate formatted data
echo "[info] Generate raw/stats for *${DataSet}"
DIR="${DataDir}"
cd ${DIR} && {
  genRaw
  genStats
  cd -;
}
