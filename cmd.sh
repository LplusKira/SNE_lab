#!/bin/bash
file="data/ml-100k/u.data"

# only want data point with rating >= 3
awk -F'\t' '{if($3 >= 3) print $0}' $file > "data/u.data.filtered"
echo "filtered data at data/u.data.filtered"

# sort by usrs
sort -t$'\t' -k1 -n "data/u.data.filtered" > "data/u.data.filtered.sorted"
echo "sorted data at data/u.data.filtered.sorted"
