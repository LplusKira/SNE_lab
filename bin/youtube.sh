#!/bin/bash
DataSet="youtube"
DataDir="../data/"

function renderData {
  echo "[info] Download + unzip dataset"
  downloadURL=$1
  staticFile=$2
  tmpDir=`mktemp -d`
  cd ${tmpDir} && { 
    curl -O ${downloadURL} 
    gzip -d ${staticFile}
    cd -
    mkdir -p ${DataDir}${DataSet}
    mv ${tmpDir}/* ${DataDir}${DataSet}
  }
  rm -rf ${tmpDir}
}

## For pulling + rendering ego-net's formatted data
downloadURL1="http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz"
downloadURL2="http://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz"
renderData ${downloadURL1} com-youtube.ungraph.txt.gz
renderData ${downloadURL2} com-youtube.all.cmty.txt.gz


## Generate formatted data
echo "[info] Generate formatted data"
DIR="${DataDir}${DataSet}/"
FeatureFile="com-youtube.all.cmty.txt"
TrainFile="com-youtube.ungraph.txt"
RevisedFeatureFE=".top10"
RevisedTrainFE=".top10"
TmpFile="row2Rank"
FilteredFE=".filtered"


### Get attr by cmt size (top 10 communities)
### ${FeatureFile} -> ${FeatureFile}${RevisedFeatureFE}
OffSet=10
awk -F'\t' '{print NR"\t"NF}' ${DIR}${FeatureFile} | sort -t$'\t' -grk2,2 | head -n${OffSet} > ${DIR}${TmpFile}
awk -F'\t' -v OffSet=${OffSet} 'FNR==NR{row2Rank[$1] = $2; next;}{if(NR - OffSet in row2Rank) print $0;}' ${DIR}${TmpFile} ${DIR}${FeatureFile} > ${DIR}${FeatureFile}${RevisedFeatureFE}
rm ${DIR}${TmpFile}

### Get the filtered trainFile
### ${TrainFile} -> ${TrainFile}${RevisedTrainFE}
Header=4
awk -F'\t' -v OffSet=${OffSet} -v Header=${Header} 'FNR==NR{for(i = 1; i <= NF; i++) u[$i] = $i; next;}{if(NR - OffSet > Header) { if($1 in u) print $1","$2;}}' ${DIR}${FeatureFile}${RevisedFeatureFE} ${DIR}${TrainFile} > ${DIR}${TrainFile}${RevisedTrainFE}

### Get the 'actual' filtered featureFile
### ${FeatureFile}${RevisedFeatureFE} -> ${FeatureFile}${RevisedFeatureFE}${FilteredFE}
python fixYoutubeFeatures.py ${DIR}${FeatureFile}${RevisedFeatureFE} ${DIR}${FeatureFile}${RevisedFeatureFE}${FilteredFE}


## Mv non-relevant files
echo "[info] Mv non-relevant files"
tmp="downloads"
mkdir -p "${DIR}${tmp}"
DontCares=`ls ${DIR} | grep -v ${FeatureFile}${RevisedFeatureFE} | grep -v ${TrainFile}${RevisedTrainFE} | grep -v ${FeatureFile}${RevisedFeatureFE} | grep -v ${FeatureFile}${RevisedFeatureFE}${FilteredFE} | grep -v ${tmp}`
cd ${DIR} && { 
  mv ${DontCares} downloads; 
  cd -; 
}
