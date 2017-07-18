#!/bin/bash

# make up yelp.data (shop, usr, rating)
#awk -F, 'FNR==NR{u2u[$1] = $1; next;}{if($2 in u2u) print $0}' usr2cnts100K yelp_academic_dataset_review.json > tmpr
#awk -F, 'FNR==NR{b2b[$1] = $1; next;}{if($3 in b2b) print $0}' biz2cnt30KFiltered tmpr > feasibleReviews.json
#awk -F, '{ print $2":"$3":"$4 }' feasibleReviews.json | awk -F':' '{ print $4"\t"$2"\t"$6; }' | sed "s/\"//g" > yelp.data

## make up user.cates (usr, cate1|cate2|...)
#awk -F, 'FNR==NR{u2u[$1] = $1; next;}{if($2 in u2u) print $2":"$3}' usr2cnts100K yelp_academic_dataset_review.json | awk -F':' '{ print $2","$4; }' > ubtmpr
#awk -F, 'FNR==NR{cates = ""; for(i = 2; i<= NF; i++) { cates = cates","$i } b2c["\""$1"\""] = cates; next;}{ print $1""b2c[$2]; }' biz2cates ubtmpr | awk -F, '{ cates = ""; for(i = 2; i<= NF; i++) { cates = cates","$i} u2c[$1] = u2c[$1]""cates; }END{ for(u in u2c) print u""u2c[u]; }' > yoman # <-- u,cates of trasaction shop's cates
#awk -F, '{for(i = 2; i<= NF; i++){ a[$i] = $i; } printf("%s", $1); for(c in a){ printf(",%s", a[c]); }; printf("\n"); delete a}' yoman > wtfff
#awk -F, 'FNR==NR{c2ind[$1] = NR; next;}{for(i = 2; i<=NF; i++){ a[c2ind[$i]] = c2ind[$i]; } printf("%s", $1); for(i = 1; i<=50; i++){ if(i in a) printf(",1,0"); else printf(",0,1");} printf("\n"); delete a;}' cates50 wtfff | sed "s/\"//g" > user.cates

# make up biz.cates (biz, its one-hot cates)
awk -F, 'FNR==NR{c2ind[$1] = NR; next;}{for(i = 2; i<=NF; i++){ a[c2ind[$i]] = c2ind[$i]; } printf("%s", $1); for(i = 1; i<=50; i++){ if(i in a) printf(",1,0"); else printf(",0,1");} printf("\n"); delete a;}' cates50 biz2cates | sed "s/\"//g" > biz2onehots
awk -F, '{print $1;}' biz2cnt30KFiltered | awk -F':' '{print $2;}' | sed "s/\"//g" > wtfff 
awk -F, 'FNR==NR{a[$1] = $1; next;}{if($1 in a) print $0}' wtfff biz2onehots > biz.cates


rm tmpr ubtmpr yoman wtfff
