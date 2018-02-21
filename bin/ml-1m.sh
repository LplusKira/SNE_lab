#!/bin/bash
source utils.sh
DataSet="ml-1m"
DataDir="../data/"

function renderData {
  echo "[info] Download + unzip dataset"
  downloadURL=$1
  zipFile="${DataSet}.zip"
  cd ${DataDir} && { 
    curl -O ${downloadURL}; 
    unzip ${zipFile}; 
    rm ${zipFile}; 
    cd -; 
  }
}

## Assert the execution place
assertCurDir $0

## Add ${DataDir} if not existent
mkdir -p ${DataDir}

## Rm 'unclean' dir
rmOldData ${DataDir}${DataSet}

## For pulling + rendering ml-100k's formatted data
downloadURL="http://files.grouplens.org/datasets/movielens/${DataSet}.zip"
renderData $downloadURL

## Generate formatted data
echo "[info] Generate formatted data"
DIR="${DataDir}${DataSet}/"
TrainFile="ratings.dat"
FeatureFile="users.dat"
FilteredFE=".filtered"
OneHottedFE=".one"
### Filter by rating
### ${TrainFile} -> ${TrainFile}${FilteredFE}
awk -F'::' '{if($3 >= 3) print $1","$2","$3;}' "${DIR}${TrainFile}" > "${DIR}${TrainFile}${FilteredFE}"

### Render one-hot-encoded attr (check ml-1m's readme for spec)
### ${FeatureFile} -> ${FeatureFile}${OneHottedFE}
python fixML1MFeatures.py "${DIR}${FeatureFile}" "${DIR}${FeatureFile}${OneHottedFE}"

### Filter encoded attr file for user in filtered rating file
### ${FeatureFile}${OneHottedFE} -> ${FeatureFile}${OneHottedFE}${FilteredFE}
awk -F',' 'FNR==NR{u[$1] = $1; next;}{if($1 in u) print $0;}' "${DIR}${TrainFile}${FilteredFE}" "${DIR}${FeatureFile}${OneHottedFE}" | sort -gk1,1 > "${DIR}${FeatureFile}${OneHottedFE}${FilteredFE}"

## Mv non-relevant files
echo "[info] Mv non-relevant files"
tmp="downloads"
mkdir -p "${DIR}${tmp}"
DontCares=`ls ${DIR} | grep -v ${TrainFile}${FilteredFE} | grep -v ${FeatureFile}${OneHottedFE} | grep -v ${FeatureFile}${OneHottedFE}${FilteredFE} | grep -v ${tmp}`
cd ${DIR} && { mv ${DontCares} downloads; cd -; }
