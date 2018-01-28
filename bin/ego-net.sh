#!/bin/bash
## For pulling + rendering ego-net's formatted data
## TODO: pulling, unzip in rendering

# Pulling
## TODO

# Rendering
## Unzip
### TODO

## Generate formatted data
DIR="../data/ego-net/"
FeatureFE=".circles"
TrainFE=".edges"
UniquserFE='.uniqusr'
RevisedFeatureFE=".u2f"
RevisedTrainFE=".u2u"
FilteredFE="filtered"
### {}${TrainFE} -> {}${TrainFE}${UniquserFE} -> {}${FeatureFE}${UniquserFE}
# TODO:
#ls ${DIR}*${TrainFE} | xargs -I{} "awk -F' ' '{a[$1] = $1;} END {for(i in a) print i}' '{}' > '{}'${UniquserFE}"

### {}${FeatureFE} -> {}${FeatureFE}${RevisedFeatureFE}
ls ${DIR}*${FeatureFE} | xargs -I{} python fixENFeatures.py {} {}${UniquserFE} {}${RevisedFeatureFE}

#### {}.${TrainFE} -> {}.${TrainFE}${RevisedTrainFE}
ls ${DIR}*${TrainFE} | xargs -I{} cp {} {}${RevisedTrainFE}

# TODO: hard-coded 'filtered'
#### {}.${FeatureFE}.u2f -> {}.${FeatureFE}.u2f.filtered
ls ${DIR}*${FeatureFE}${RevisedFeatureFE} | xargs -I{} sh -c 'cat "$1" | cut -d, -f1-35 > "$1".filtered' -- {}
#### {}.${TrainFE}.u2u -> {}.${TrainFE}.u2u.filtered
