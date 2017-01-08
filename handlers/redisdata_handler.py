import sys
sys.path.insert(0, '../')

import json
from config import none_dmplabels

class redisdata_handler:
    def __init__(self, for_module_cliredis, update_to_cliredises):
        self.cli = for_module_cliredis
        self.clis =  update_to_cliredises

    # methods for for_module_cliredis
    def get_usr_recs(self, subscriberid):
        return self.cli.hgetall('usrRecords_' + subscriberid)

    def del_outdated_records(self, subscriberid, outdated_t_recs):
        if outdated_t_recs:
            self.cli.hdel( 'usrRecords_' + subscriberid, *map(lambda t_rec: t_rec['unixtime'], outdated_t_recs) )

    def init_usr_clickhistory(self, subscriberid):
        self.cli.hmset( 'usrClicks_' + subscriberid, {'click':0, 'nonclick': 0} )

    # add usr_rec only when `for-agg` data are ready
    def add_usr_recs(self, subscriberid, cur_rec_time, imp):
        click = imp['clickOrNot']
        reqs = imp['bidrequests']
        if reqs:
            iabcats_list = map(lambda req: req['iabcats'], reqs)
            iabcats_cnts = reduce(lambda r1, r2: [sum(v) for v in zip(r1, r2)], iabcats_list)
            iablabels_str = ','.join(map(lambda v: str(v), iabcats_cnts))

            update_rec = ';'.join([
                iablabels_str,
                str(click),
            ])
            self.cli.hmset( 'usrRecords_' + subscriberid, {str(cur_rec_time): update_rec} )

    def update_usr_clickhistory(self, subscriberid, click):
        if(click):
            self.cli.hincrby('usrClicks_' + subscriberid, 'click', amount=1)
        else:
            self.cli.hincrby('usrClicks_' + subscriberid, 'nonclick', amount=1)

    def get_usr_ctr(self, subscriberid):
        click_rec = self.cli.hgetall( 'usrClicks_' + subscriberid )
        click     = float(click_rec['click'])
        nonclick  = float(click_rec['nonclick'])
        return click/ (click + nonclick)

    def get_ad_features(self, adid):
        adProfile = self.cli.hgetall('adProfile_' + adid)
	features = []

	# adProfile could be {} || adProfile has no aggfeatures
	if not adProfile or not 'aggfeatures' in adProfile:
            init = self.cli.hget('adid_' + adid, 'features')
	    features = json.loads( init ) if init else []
	else:
	    features = adProfile['aggfeatures']
	    features = map(lambda v: int(v), features.split(','))

        return features

    def get_dmplabels(self, subscriberid):
        try:
            # assume dmplabels == '0,1,0,...'
            dmplabels = self.cli.hget('usrProfile_' + subscriberid, 'dmplabels')
            return dmplabels if dmplabels else none_dmplabels
        except:
            return none_dmplabels

    # methods for update_to_cliredises
    def update_usr_profile(self, subscriberid, aggfeatures):
        for c in self.clis:
            c.hmset( 'usrProfile_' + subscriberid, {'aggfeatures': aggfeatures} )
  
    def update_ad_Profile(self, adid, features):
        for c in self.clis:
            c.hmset( 'adProfile_' + adid, {'aggfeatures': features} )
