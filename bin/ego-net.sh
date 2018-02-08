#!/bin/bash
source utils.sh
DataSet="ego-net"
DataDir="../data/"

function renderData {
  echo "[info] Download + unzip dataset"
  downloadURL=$1
  gzFile="${DataSet}.tar.gz"
  tmpDir=`mktemp -d`
  cd ${tmpDir} && { 
    curl -o ${gzFile} ${downloadURL}; 
    tar -xzvf ${gzFile}; 
    cd -; 
    mv "${tmpDir}/`ls ${tmpDir} | grep -v ${gzFile}`" ${DataDir}${DataSet}; 
  }
  rm -rf ${tmpDir}
}

## Assert the execution place
assertCurDir $0

## Rm 'unclean' dir
echo "[info] Remove ${DataDir}${DataSet}"
rm -rf ${DataDir}${DataSet}

## For pulling + rendering ego-net's formatted data
downloadURL="http://snap.stanford.edu/data/facebook.tar.gz"
renderData $downloadURL


## Generate formatted data
echo "[info] Generate formatted data"
DIR="${DataDir}${DataSet}/"
FeatureFE=".circles"
TrainFE=".edges"
UniquserFE='.uniqusr'
RevisedFeatureFE=".u2f"
RevisedTrainFE=".u2u"
FilteredFE="filtered"
### Get uniqUsr from tranfile; rename to feature.uniq
### {}${TrainFE} -> {}${TrainFE}${UniquserFE} -> {}${FeatureFE}${UniquserFE}
ls ${DIR}*${TrainFE} | xargs -I{} sh -c 'cat "$1" | cut -d " " -f 1 | uniq > "$1".uniqusr' -- {}
cd ${DIR} && {
  for f in *${TrainFE}${UniquserFE}; do mv "$f" "${f%${TrainFE}${UniquserFE}}${FeatureFE}${UniquserFE}"; done
  cd -;
}

### Encode each user's attributes
### {}${FeatureFE} -> {}${FeatureFE}${RevisedFeatureFE}
ls ${DIR}*${FeatureFE} | xargs -I{} python fixENFeatures.py {} {}${UniquserFE} {}${RevisedFeatureFE}

### Revise trainfile
### {}.${TrainFE} -> {}.${TrainFE}${RevisedTrainFE}
#ls *${TrainFE} | xargs -I{} cp {} {}${RevisedTrainFE}
cd ${DIR} && {
  for f in *${TrainFE}; do 
    awk -F' ' '{print $1","$2;}' "$f" > "$f${RevisedTrainFE}"
  done
  cd -;
}

#### Filter 1st 35 cols (17 attributes)
#### {}.${FeatureFE}${RevisedFeatureFE} -> {}.${FeatureFE}${RevisedFeatureFE}${FilteredFE}
ls ${DIR}*${FeatureFE}${RevisedFeatureFE} | xargs -I{} sh -c 'cat "$1" | cut -d, -f1-35 > "$1".filtered' -- {}

## Mv non-relevant files
echo "[info] Mv non-relevant files"
tmp="downloads"
mkdir -p "${DIR}${tmp}"
DontCares=`ls ${DIR} | grep -v ${FeatureFE}${UniquserFE} | grep -v ${FeatureFE}${RevisedFeatureFE} | grep -v ${TrainFE}${RevisedTrainFE} | grep -v ${FeatureFE}${RevisedFeatureFE}${FilteredFE} | grep -v ${tmp}`
cd ${DIR} && { 
  mv ${DontCares} downloads; 
  cd -; 
}
