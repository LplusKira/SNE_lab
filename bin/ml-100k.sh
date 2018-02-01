#!/bin/bash
DataSet="ml-100k"
DataDir="../data/"

function renderData {
  echo "[info] Download + unzip dataset"
  downloadURL=$1
  zipFile="${DataSet}.zip"
  cd ${DataDir} && { curl -O ${downloadURL} ; unzip ${zipFile}; cd -; }
}

## For pulling + rendering ml-100k's formatted data
downloadURL="http://files.grouplens.org/datasets/movielens/${DataSet}.zip"
renderData $downloadURL

## Generate formatted data
echo "[info] Generate formatted data"
DIR="${DataDir}${DataSet}/"
Prefix="u"
TrainFE=".data"
FeatureFE=".user"
FilteredFE=".filtered"
OneHottedFE=".one"
### Filter by rating
### ${Prefix}${TrainFE} -> ${Prefix}${TrainFE}${FilteredFE}
awk -F'\t' '{if($3 >= 3) print $1","$2","$3;}' "${DIR}${Prefix}${TrainFE}" > "${DIR}${Prefix}${TrainFE}${FilteredFE}"

### Render one-hot-encoded attr (check ml-100k's readme for spec)
### ${Prefix}${FeatureFE} -> ${Prefix}${FeatureFE}${OneHottedFE}
python fixML100KFeatures.py "${DIR}${Prefix}${FeatureFE}" "${DIR}${Prefix}${FeatureFE}${OneHottedFE}"

### Filter encoded attr file for user in filtered rating file
### ${Prefix}${FeatureFE}${OneHottedFE} -> ${Prefix}${FeatureFE}${OneHottedFE}${FilteredFE}
awk -F',' 'FNR==NR{u[$1] = $1; next;}{if($1 in u) print $0;}' "${DIR}${Prefix}${TrainFE}${FilteredFE}" "${DIR}${Prefix}${FeatureFE}${OneHottedFE}" | sort -gk1,1 > "${DIR}${Prefix}${FeatureFE}${OneHottedFE}${FilteredFE}"

## Mv non-relevant files
echo "[info] Mv non-relevant files"
tmp="downloads"
mkdir -p "${DIR}${tmp}"
DontCares=`ls ${DIR} | grep -v ${Prefix}${TrainFE}${FilteredFE} | grep -v ${Prefix}${FeatureFE}${OneHottedFE} | grep -v ${Prefix}${FeatureFE}${OneHottedFE}${FilteredFE} | grep -v ${tmp}`
cd ${DIR} && { mv ${DontCares} downloads; cd -; }
