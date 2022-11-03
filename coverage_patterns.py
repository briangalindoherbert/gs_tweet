# encoding=utf-8
"""
functions to trace threads of tweets through replies, retweets, and quote tweets
and add metrics for each entry
"""
import re

def build_linked_trace(twful: dict):
    """
    with single pass through dataset, get core_ids plus 2nd level of reply, qt, or rts
    if applicable
    :param twful: full dict of dict with all tweets and fields
    :return: dict of dict - recursive trace through originating tweets
    """
    tmptrace: dict = twful.copy()
    tmpdct: dict = {k: trace_hit(k, gendct={'gen': 'prime'}, twfl=twful) for k in tmptrace}

    print(f" trace_from_firstpass added {len(tmpdct)} first level records")

    return tmpdct

def trace_hit(tracek: str, gendct: dict, twfl: dict):
    """
    atomic parse and fill for id's and metrics
    :param tracek: the tweet ID value to trace
    :param gendct: passes in source designator {'gen':'rply'} for ID passed in parm 0
    :param twfl: set of all tweet ID's in dataset
    :return: dict with fave, qrr, compound sentiment plus any next rply, rt or qt ID's
    """
    retdct: dict = {}
    if tracek in twfl:
        tmpdct: dict = twfl[tracek]
        retdct: dict = {'quot':tmpdct['quot'], 'rtwt':tmpdct['rtwt'], 'rply':tmpdct['rply'],
                        'fave': tmpdct['fave'], 'comp':tmpdct['compound']} | gendct
        if 'rply_id' in tmpdct and re.search(r"[0-9]{15,}", tmpdct['rply_id']):
            retdct |= {tmpdct['rply_id']: trace_hit(tmpdct['rply_id'], {'gen':'rply_id'}, twfl)}
        if 'rt_id' in tmpdct and re.search(r"[0-9]{15,}", tmpdct['rt_id']):
            retdct |= {tmpdct['rt_id']: trace_hit(tmpdct['rt_id'], {'gen':'rt_id'}, twfl)}
        if 'qt_id' in tmpdct and re.search(r"[0-9]{15,}", tmpdct['qt_id']):
            retdct |= {tmpdct['qt_id']: trace_hit(tmpdct['qt_id'], {'gen':'qt_id'}, twfl)}
    else:
        retdct |= gendct

    return retdct

def has_metrics(mdct: dict):
    """
    atomic fx to verify presence of the 3 core metrics- qrr, fave, comp.
    :param mdct: a value dict for a particular trace key
    :return: True or False
    """
    return 'comp' in mdct and 'quot' in mdct and 'rtwt' in mdct and 'rply' in mdct and 'fave' in mdct

def get_keyval_pairs(nextd: dict):
    """
    atomic fx to identify any keys in the value dict for a higher level key
    :param nextd: a value dict for a particular trace key
    :return: a dict made up of zero-many key:value pairs
    """
    return sum(1 + get_keyval_pairs(v1) for k1, v1 in nextd.items() if re.search(r"[0-9]{15,}",
                                                                                 k1) and isinstance(v1, dict))

def get_trace_depth(trace: dict):
    """
    get the longest traces in the dataset.
    :param trace: the trace created from build_linked_trace
    :return: dict with number of keys for each 'level zero' key
    """
    traced: dict = {k0: get_keyval_pairs(v0) for k0, v0 in trace.items()}
    traced = {k: traced[k] for k in sorted(traced, key=lambda x: traced.get(x), reverse=True)}
    longest: int = max(traced[x] for x in traced)
    average: int = round(sum(traced[x] for x in traced) / len(traced), ndigits=2)
    print("\n    get_trace_depth calculates length of each key trace")
    print(f"    average keys in each trace: {average}")
    print(f"    longest trace: {longest}")
    print(f"    total traces by Tweet ID: {len(traced)}")

    return traced

def count_missing_types(trace: dict, cnt: dict, lvl0: bool=False):
    """
    after get missing traces, this counts the types of ID's in the result set
    :param trace: a trace of linked tweets, IDs and metrics at each level
    :param cnt: a dict for passing values recursively
    :param lvl0: recursion flag to be set true for first level call only
    :return:
    """
    if lvl0 and not cnt:
        cnt = {'rt': 0, 'rp': 0, 'qt': 0, 'prime': 0}
    for v in trace.values():
        if isinstance(v, dict):
            if v['gen'] == 'prime':
                cnt['prime'] += 1
            elif v['gen'] == 'rt_id':
                cnt['rt'] += 1
            elif v['gen'] == 'qt_id':
                cnt['qt'] += 1
            elif v['gen'] == 'rply_id':
                cnt['rp'] += 1
            for v1 in v.values():
                if isinstance(v1, dict):
                    count_missing_types(v1, cnt=cnt, lvl0=False)

    if lvl0:
        print(f"\n  count_missing_types found {cnt['prime']} Tweet id's'")
        print(f"  {cnt['rt']} retweets, {cnt['qt']} quote tweets, and {cnt['rp']} replies")

    return

def get_keys_missing_metrics(trc: dict, lvl0: bool=False):
    """
    look through trace dataset for missing metrics and identify by ID
    :param trc: the trace dict of tweets
    :param lvl0: boolean allowing id of lowest level fx call
    :return: trace_get dictionary
    """
    miss_keys: list = []

    for k, v in trc.items():
        if isinstance(v, dict) and not has_metrics(v):
            miss_keys.append(k)
        if newkv := get_keyval_pairs(v):

            miss_keys.extend(get_keys_missing_metrics(newkv))

    if lvl0:
        print("  find_trace_metrics- identifies trace keys that lack metrics")
        print(f"      identified {len(miss_keys)}")

    return miss_keys

def find_fill_2ndary(trc: dict, secnd: dict, fillname: str='qt_id'):
    """
    atomic utility to populate the metrics given a tweet ID
    'gen' values can be 'src' for metrics from Tweet source, or 'scnd' if filled
    from a 'rt', 'qt', or 'rp'.
    :param trc: dict of dict- key-value pairs for traces
    :param secnd: dataset with derived metrics from rt, qt and replies
    :return: dict with keys: 'comp', 'qrr', 'fave' and 'gen'
    """
    retdct: dict = {}
    for k, v in trc.items():
        if isinstance(v, dict) and k in secnd and not has_metrics(v):
            v |= {'quot': secnd[k]['quot'], 'rtwt': secnd[k]['rtwt'], 'fave': secnd[k]['fave'],
                  'rply': secnd[k]['rply'], 'comp': secnd[k]['compound'], 'gen': fillname}
            v |= find_fill_2ndary(v, secnd, fillname=fillname)

        retdct[k] = v

    return retdct

def multi_enrich_second(twful: dict, id2nd: set):
    """
    look up metrics for ID's with missing metrics from prior step
    :param twful: the full list of dict tweet dataset
    :param id2nd: set of all secondary id's (rply and qt ids not in primary ids)
    :return: enriched dict of tweets derived from RTs, QTs and Replies
    """
    enriched: dict = {}
    found: int = 0
    for k, v in twful.items():
        tmpdct: dict = {}
        for typ in ['qt', 'rt', 'rply']:
            seek: str = f"{typ}_id"
            if seek in v and v[seek] in id2nd:
                thiskey: str = v[seek]
                found += 1
                if f"{typ}_compound" in v:
                    tmpdct |= {'comp': v[f"{typ}_compound"], 'gen': seek}
                else:
                    tmpdct |= {'comp': None, 'gen': seek}

                if f"{typ}_quot" in v:
                    tmpdct['quot'] = v[f"{typ}_quot"]
                elif "rt_quot" in v:
                    tmpdct['quot'] = v["rt_quot"]

                if f"{typ}_rply" in v:
                    tmpdct['rply'] = v[f"{typ}_rply"]
                elif "rt_rply" in v:
                    tmpdct['rply'] = v["rt_rply"]

                if f"{typ}_rtwt" in v:
                    tmpdct['rtwt'] = v[f"{typ}_rtwt"]
                elif "rt_rtwt" in v:
                    tmpdct['rtwt'] = v["rt_rtwt"]

                if f"{typ}_fave" in v:
                    tmpdct['fave'] = v[f"{typ}_fave"]
                elif "rt_fave" in v:
                    tmpdct['fave'] = v["rt_fave"]
                else:
                    tmpdct['fave'] = 0

                enriched[thiskey] = tmpdct

    print("\n  enrich all ID's in dataset")
    print(f"          found {found} secondary ID's")
    print(f"          in {len(twful)} total records")

    return enriched

def enrich_secondary_ids(twful: dict, id2nd: set, typ: str='qt'):
    """
    look up metrics for ID's with missing metrics from prior step
    :param twful: the full list of dict tweet dataset
    :param id2nd: set of all secondary id's (rply and qt ids not in primary ids)
    :param typ: build either 'qt', 'rply' or 'rt' secondary sources
    :return: enriched dict of tweets derived from RTs, QTs and Replies
    """
    enriched: dict = {}
    found: int = 0
    for k, v in twful.items():
        seek: str = f"{typ}_id"
        if seek in v and v[seek] in id2nd:
            found += 1
            tmpdct: dict = {'gen': seek}
            if f"{typ}_compound" in v:
                tmpdct |= {'comp': v[f"{typ}_compound"]}
            else:
                tmpdct |= {'comp': None}

            if f"{typ}_quot" in v:
                tmpdct['quot'] = v[f"{typ}_quot"]
            elif "rt_quot" in v:
                tmpdct['quot'] = v["rt_quot"]

            if f"{typ}_rply" in v:
                tmpdct['rply'] = v[f"{typ}_rply"]
            elif "rt_rply" in v:
                tmpdct['rply'] = v["rt_rply"]

            if f"{typ}_rtwt" in v:
                tmpdct['rtwt'] = v[f"{typ}_rtwt"]
            elif "rt_rtwt" in v:
                tmpdct['rtwt'] = v["rt_rtwt"]

            if f"{typ}_fave" in v:
                tmpdct['fave'] = v[f"{typ}_fave"]
            elif "rt_fave" in v:
                tmpdct['fave'] = v["rt_fave"]
            else:
                tmpdct['fave'] = 0

        enriched[v[seek]] = tmpdct

    print("\n  enrich all ID's in dataset")
    print(f"          found {found} secondary ID's")
    print(f"          in {len(twful)} total records")

    return enriched

def populate_missing(getd: dict, traced: dict):
    """
    add fave, qrr, comp metrics to trace items
    :param getd: dict of metrics from gets
    :param traced: dict of values from trace
    :return:
    """
    if 'fave' not in traced or getd['fave'] > traced['fave']:
        traced['fave'] = getd['fave']
    if 'quot' not in traced or getd['quot'] > traced['quot']:
        traced['quot'] = getd['quot']
    if 'rtwt' not in traced or getd['rtwt'] > traced['rtwt']:
        traced['rtwt'] = getd['rtwt']
    if 'rply' not in traced or getd['rply'] > traced['rply']:
        traced['rply'] = getd['rply']
    if 'comp' not in traced or getd['comp'] > traced['comp']:
        traced['comp'] = getd['comp']
    traced['gen'] = getd['gen']

    return traced

def fill_trace_fromgets(gets: dict, twtrace: dict):
    """
    populate trace dataset using found metrics, 1st level should be populated, so start at 2nd level
    :param gets:
    :param twtrace:
    :return:
    """
    metricsget: dict = {'lvl1': 0, 'lvl2': 0, 'lvl3': 0, 'lvl4': 0}
    filldct: dict = {}
    for k, v in twtrace.items():
        if k in gets:
            filldct[k] = populate_missing(gets[k], twtrace[k])
            metricsget['lvl1'] += 1
        else:
            filldct[k] = twtrace[k]
        for k1 in v:
            if re.search(r"[0-9]{15,}", k1) and k1 in gets:
                filldct[k1] = populate_missing(gets[k1],twtrace[k1])
                metricsget['lvl2'] += 1
            else:
                filldct[k1] = twtrace[k1]
            for k2 in twtrace[k1]:
                if re.search(r"[0-9]{15,}", k2) and k2 in gets:
                    filldct[k2] = populate_missing(gets[k2], twtrace[k2])
                    metricsget['lvl3'] += 1
                else:
                    filldct[k2] = twtrace[k2]
                for k3 in twtrace[k2]:
                    if re.search(r"[0-9]{15,}", k3) and k3 in gets:
                        filldct[k3] = populate_missing(gets[k3], twtrace[k3])
                        metricsget['lvl4'] += 1
                    else:
                        filldct[k3] = twtrace[k3]

    print(f"\n    fill_trace_fromgets: searching {len(twtrace)}")
    print(f"      {metricsget['lvl1']} first level metrics added")
    print(f"      {metricsget['lvl2']} second level metrics added")
    print(f"      {metricsget['lvl3']} third level metrics added")
    print(f"      {metricsget['lvl4']} fourth level metrics added")

    return filldct

def apply_filter(twx: dict, field: str='fave', fval: int=2):
    """
    Fx to filter tweet records by the value of a metrics field,
    such as 'Liked count is at least 2'
    :param twx:
    :param field:
    :param fval:
    :return:
    """
    if field in twx:
        if twx[field] >= fval:
            return twx
    else:
        print(f"ERROR:  apply_filter {field} is not a valid filter field!")

    return None

def filter_select_rply_qt(twl: dict, favemin: int=1):
    """
    for reply or quote tweets that meet importance criteria, identify if we have
    or don't have the originating tweet.  use list of missing ID's to prioritize
    subsequent GET's from Twitter full search endpoint.
    :param twl: dict of dict of tweets
    :param favemin: minimum acceptable count of Likes (Favorites)
    :return:
    """
    twcopy: dict = twl.copy()
    retfound: dict = {}
    retnotfound: dict = {}

    for k, v in twcopy.items():
        if apply_filter(v, field='fave', fval=favemin):
            if 'rply_id' in v:
                if v['rply_id'] in twl:
                    retfound[k] |= {'source': 'rply_id'}
                else:
                    retnotfound[k] |= {'source': 'rply_id'}
            if 'qt_id' in v:
                if v['qt_id'] in twl:
                    retfound[k] |= {'source': 'qt_id'}
                else:
                    retnotfound[k] |= {'source': 'qt_id'}
            if 'rt_id' in v:
                if v['rt_id'] in twl:
                    retfound[k] |= {'source': 'rt_id'}
                else:
                    retnotfound[k] |= {'source': 'rt_id'}
            if retfound[k]:
                retfound[k] |= v
            if retnotfound[k]:
                retnotfound[k] |= v

    return retfound, retnotfound

def find_andcount(txt: str, findtxt: list, minf: int=3):
    """
    utility function using re to look for text and return boolean if minf matches was found
    :param txt: str text to search
    :param findtxt: list of words or phrases to search for in txt
    :param minf: minimum number of matches needed
    :return: boolean for match
    """
    foundw: bool = False
    tot_found: int = 0
    for tmpw in findtxt:
        tot_found += len(re.findall(tmpw, txt))
        if tot_found >= minf:
            foundw = True
            continue

    return foundw

def filter_set(twl: list, minwrd: int=2, minphrase: int=1):
    """
    eliminates tweets with many junk words or repeated words
    :param twl: list of dict of tweet
    :param minwrd: minimum num of positive words from list
    :param minphrase: minimum occurrence needed for phrase
    :return list of filtered tweets
    """
    print(f"\n  filter_set being applied to {len(twl)} tweets")
    good_words: list = ['privilege', 'wellbeing', 'resonating', 'disengagement', 'family',
                        'empowerment', 'prominent', 'creativity', 'managers', 'burnout',
                        'packages', 'collaboration', 'onboarding', 'millennials', 'overwork',
                        'gen x', 'genx', 'gen z', 'genz', 'boundaries', 'laziness',
                        'workday', 'privilege',
                        ]
    good_phrases: list = ['shutting down', 'turning off', 'shutting down', 'end workday',
                          'worklife balance', 'setting boundaries', 'work harder',
                          'build relationships', 'quality candidates', 'rejecting overwork',
                          'generational divide', 'best employees', 'quality staff'
                          ]
    tmplst: list = []
    twl_copy: list = twl.copy()
    twl_len: int = len(twl)
    wrds_ct: int = 0
    phrase_ct: int = 0
    for tw in twl_copy:
        # check for overly vulgar or ignorant tweets...
        if find_andcount(tw['text'], good_words, minf=minwrd):
            wrds_ct += 1
            if find_andcount(tw['text'], good_phrases, minf=minphrase):
                phrase_ct += 1
                tmplst.append(tw)

    if len(tmplst) > 1:
        print(f"\n  filter_set:    {twl_len} candidate tweets in dataset")
        print(f"                  {wrds_ct} passed word test")
        print(f"                  {phrase_ct} passed phrase test")
        print("                  -----------")
        print(f"      ended with  {len(tmplst)} tweets")

    return tmplst

def identify_conversations(twd, qtid: set, rpid: set, rtid: set):
    """
    identify conversation IDs and how many associated Tweets for each
    :param twd: dict of dict for Tweet dataset
    :param qtid: set of quote tweet originating IDs
    :param rpid: set of originating IDs for replies
    :param rtid: set of originating IDs for retweets
    :return:
    """
    conv: list = [x['conversation'] for x in twd.values() if 'conversation' in x]
    conv: set = set(conv)

    conv_N_qt: set = conv.intersection(qtid)
    conv_N_rp: set = conv.intersection(rpid)
    conv_N_rt: set = conv.intersection(rtid)
    qtrtrp_conv: set = conv_N_qt.union(conv_N_rp).union(conv_N_rt)
    conv_noid: set = qtrtrp_conv.difference(set(twd.keys()))

    return set(sorted(conv_noid))

def check_tweet_activity_coverage(twful: dict, twdc: dict, rpdc: dict, qtdc: dict, rtdc: dict):
    """
    go through list of dict of Tweets, get the reply and quote count for each, then
    compare it with how many reply or quote records we have for that ID.
    Identifies total activity (potential tweet records) across reply, quote, retweet, and
    original.  Also identifies records in dataset across the 4 source types.
    :param twl: dict of dict of tweet dataset
    :param twdc: Dict of key:ID, value: number of records
    :param rpdc:
    :param qtdc:
    :param rtdc:
    :return: dict with replies and quotes, and number of records in dataset by ID
    """
    primary_metrics: list = ['quot', 'rply', 'rtwt']
    idtyp_dct: dict = {"primary": twdc, "have replies": rpdc, "have quotes": qtdc,
                       "have retweets": rtdc}
    outdct: dict = {}
    missing_counter: int = 0
    for k, v in twful.items():
        tmpdct: dict = {'typ': 'id', 'tdate': v['tdate']}
        activity: int = 1
        for x in primary_metrics:
            if x in v:
                tmpdct |= {x: v[x]}
                activity += v[x]

        rec_count: int = 0
        for label, iddct in idtyp_dct.items():
            if k in iddct:
                tmpdct[label] = iddct[k]
                rec_count += iddct[k]

        tmpdct['twitter_recs'] = activity
        tmpdct['dataset_recs'] = rec_count
        tmpdct['online_less_ds'] = int(activity - rec_count)
        missing_counter += int(activity - rec_count)
        if 'conversation' in v:
            tmpdct['conv'] = v['conversation']
        # only writing records with sufficient number of distribution actions
        if k in outdct:
            outdct[k] |= tmpdct
        else:
            outdct[k] = tmpdct

    print(f"  check_tweet_activity_coverage wrote {len(outdct):,} records")
    print(f"      counted {missing_counter:,} tweets missing from dataset")

    return outdct

def check_rpqt_coverage(twful: dict, twdc: dict, rpdc: dict, qtdc: dict, rtdc: dict):
    """
    For Replies, Retweets, and Quoted Tweets in our Tweet dataset, capture how many RTs, QTs,
    and Replies there were on the Originating Tweet.
    Also count how many are in our dataset by type (reply, quote, retweet, original).
    :param twful: dict of dict of tweets
    :param twdc: dict of all tweet IDs and number of records
    :param rpdc:  dict of all reply IDs and number of records
    :param rtdc: dict of all RT IDs and number of records
    :param qtdc: all quote tweet IDs and number of records
    :return: filtered tweets with reply/quote counts versus actual records in dataset
    """
    redist_all: list = ['qt', 'rt', 'rply']
    metrics_suffix: list = ['_quot', '_rply', '_rtwt']
    # get a list of all the possible tweet redistribution metrics
    # redist_all = [f"{a}{b}" for a, b in zip(redist_all, metrics_suffix)]
    outdct: dict = {}
    ops_data: dict = {}
    for k, v in twful.items():
        for disttyp in redist_all:
            idtyp: str = f"{disttyp}_id"
            if idtyp in v:
                if idtyp in ops_data:
                    ops_data[idtyp] += 1
                else:
                    ops_data[idtyp] = 1

                tmpkey: str = v[idtyp]
                redst: dict = check_redist_metrics(v, retyp=disttyp)
                tot_activity: int = 1 + sum(redst.values())
                tmpdct: dict = {'twitter_recs': tot_activity, 'typ': idtyp} | redst
                rec_count: int = 0
                if tmpkey in twdc:
                    tmpdct |= {'recs': twdc[tmpkey]}
                    rec_count += twdc[tmpkey]
                if tmpkey in rpdc:
                    tmpdct |= {'rp_recs': rpdc[tmpkey]}
                    rec_count += rpdc[tmpkey]
                if tmpkey in rtdc:
                    tmpdct |= {'rt_recs': rtdc[tmpkey]}
                    rec_count += rtdc[tmpkey]
                if tmpkey in qtdc:
                    tmpdct |= {'qt_recs': qtdc[tmpkey]}
                    rec_count += qtdc[tmpkey]
                # if we've captured metrics for this RT, QT, or Reply, save it
                tmpdct['dataset_recs'] = rec_count
                tmpdct['online_less_ds'] = tot_activity - rec_count
                if 'conversation' in v:
                    tmpdct['conv'] = v['conversation']
                if tmpkey in outdct:
                    outdct[tmpkey] |= tmpdct
                else:
                    outdct[tmpkey] = tmpdct

    print("\n    check_redistro_coverage")
    print(f"        wrote {len(outdct)} records for Originating Tweets")
    print(f"        analyzed {sum(ops_data.values()):,} QT, RT, and Replies in Dataset")
    print("     contents of processing data below: ")
    print(ops_data)

    return outdct

def check_redist_metrics(vald: dict, retyp: str='qt'):
    """
    refactored function from Tweet coverage functions in this module, for a reply,
    quote, or retweet, check for existence of 'derived' or secondary metrics using
    all 3 possible types (if Originating Tweet is not in dataset).
    :param vald: value dictionary for a single Tweet from dataset
    :param tkey: the Tweet ID for the Originating Tweet
    :return:
    """
    distdct: dict = {}
    # for retyp in ['qt', 'rt', 'rply']:
    for suffix in ['_quot', '_rply', '_rtwt']:
        seektyp: str = f"{retyp}{suffix}"
        if seektyp in vald:
            distdct |= {seektyp: vald[seektyp]}

    return distdct

def filter_match_coverage_sets(prime: dict, scnd: dict):
    """
    match up the dicts for Tweet redistribution coverage, on ID, conversation ID,
    using set functions for intersection and difference.
    :param prime:
    :param scnd:
    :return:
    """
    primelst: list = []
    for k, v in prime.items():
        if 'conv' in v:
            primelst.append(k)

    scndlst: list = []
    for k, v in scnd.items():
        if 'conv' in v:
            scndlst.append(k)

    combinedset: set = set(scndlst).intersection(set(primelst))

    return combinedset

def update_scnd_from_primary(twful: dict, scndcvr: dict):
    """
    run this after running check_rpqt_coverage to update the direct reply, retweet,
    or quote counts for found primary ID's
    :param twful:
    :param scndcvr:
    :return:
    """
    primary_metrics: list = ['quot', 'rply', 'rtwt']
    outdct: dict = {}
    prime_upd: int = 0
    for k, v in scndcvr.items():
        if k in twful:
            primedct: dict = twful[k]
            tmpdct: dict = {'typ': 'prime'}
            prime_upd += 1
            for xmetric in primary_metrics:
                if xmetric in primedct:
                    tmpdct |= {xmetric: primedct[xmetric]}
            v |= tmpdct

        outdct[k] = v

    print(f"\n    updated {prime_upd} tweets from primary tweet records")

    return outdct

def get_best_metric_values(scnd: dict):
    """
    from output from check_rpqt_coverage, identify best metrics for quot, rtwt, and rply,
    this allows a single calc of metrics vs Tweets in the dataset.
    :param scnd:
    :return:
    """
    metric_typs = ['quot', 'rtwt', 'rply']
    dist_typs = ['qt', 'rt', 'rply']
    outdct: dict = {}
    metrics_changed: int = 0
    for k, v in scnd.items():
        best: dict = {}
        for x in metric_typs:
            best[x] = v[x] if x in v else 0
            for y in dist_typs:
                thistyp = f"{y}_{x}"
                if thistyp in v and v[thistyp] > best[x]:
                    metrics_changed += 1
                    best[x] = v[thistyp]
                    v[x] = best[x]

        outdct[k] = v

    print("\n    get_best_metrics finding most up to date source")
    print(f"        changed {metrics_changed} quote-retweet-reply metrics")
    print(f"        for     {len(outdct)} tweet records\n")

    return outdct

def split_influential_into_missing_ornot(topids: list, twdx: set):
    """
    apply list of most influential tweets from coverage functions to the
    set of all Tweet IDs in the dataset, and split into a list of those
    we have in dataset and those that are missing.
    :param topids: list of IDs from rpqt_coverage
    :param twdx: set of Tweet IDs in dataset
    :return: list of missing, list of not missing
    """
    toppop_missing: list = []
    toppop_havetweet: list = []
    ct_missing: int = 0
    total_recs: int = 0
    for xid in topids:
        total_recs += 1
        if xid not in twdx:
            ct_missing += 1
            toppop_missing.append(xid)
        else:
            toppop_havetweet.append(xid)

    print(f"  out of {total_recs} most distributed Tweets, {total_recs - ct_missing} are in the dataset")

    return toppop_missing, toppop_havetweet

def get_topgap_quote_reply(twd: dict, twdx: dict, qtdx: dict, rpdx: dict):
    """
    look through coverage set- identify diff between our quote count
    and quote metrics, as well as for replies.
    write top gaps to a list to get from the API.
    :param twd:
    :param twdx:
    :param qtdx:
    :param rtdx:
    :return:
    """
    gap_dct: dict = {}
    for k, v in twd.items():
        if 'quot' in v:
            if 'qt_recs' in v:
                qgap: int = v['quot'] - v['qt_recs']

    return

def add_missing_original_recs(twful: dict):
    """
    if we have a quoted or reply record with info on the orignal tweet,
    we can add a record for it.
    :param twful: full dict of dict with all tweets and fields
    :return: dict of dict - recursive trace through originating tweets
    """
    tmptrace: dict = twful.copy()
    tmpdct: dict = {k: trace_hit(k, gendct={'gen': 'prime'}, twfl=twful) for k in tmptrace}

    print(f" trace_from_firstpass added {len(tmpdct)} first level records")

    return tmpdct

def report_dataset_tweet_types(twl: list, rtc: dict, qtc: dict, rpc: dict):
    """
    looks at the dataset after parsing and duplicate filtering has been run,
    and reports on it's state
    :param twl: the list of dict Tweet dataset
    :param rtc: dict of retweet count by originating ID
    :param qtc: dict of quote tweet count by originating ID
    :param rpc: dict of reply count by originating ID
    :return:
    """
    tw_noid: int = 0
    has_rt: int = 0
    has_qt: int = 0
    has_rply: int = 0
    allids: dict = {}
    for x in twl:
        if isinstance(x, dict):
            if 'id' in x:
                if x['id'] in allids:
                    allids[x['id']] |= {'prime': True}
                else:
                    allids[x['id']] = {'prime': True, 'retweet': False, 'quote': False, 'reply': False}
            if 'rt_id' in x and len(x['rt_id']) > 5:
                if x['rt_id'] in allids:
                    allids[x['rt_id']] |= {'retweet': True}
                else:
                    allids[x['rt_id']] = {'prime': False, 'retweet': True, 'quote': False, 'reply': False}
            if 'qt_id' in x and len(x['qt_id']) > 5:
                if x['qt_id'] in allids:
                    allids[x['qt_id']] |= {'quote': True}
                else:
                    allids[x['qt_id']] = {'prime': False, 'retweet': False, 'quote': True, 'reply': False}
            if 'rply_id' in x and len(x['rply_id']) > 5:
                if x['rply_id'] in allids:
                    allids[x['rply_id']] |= {'reply': True}
                else:
                    allids[x['rply_id']] = {'prime': False, 'retweet': False, 'quote': False, 'reply': True}

    for k, v in allids.items():
        if not v['prime']:
            tw_noid += 1
        if v['retweet']:
            has_rt += 1
        if v['quote']:
            has_qt += 1
        if v['reply']:
            has_rply += 1

    print("\n  Twitter dataset after parse and filter:")
    print(f"   {len(allids):,} unique Tweet ids")
    print(f"          {has_rt} have retweets in dataset")
    print(f"          {has_rply} have replies in dataset")
    print(f"          {has_qt} have quote tweets in dataset")
    print(f"\n   {sum(rtc.values())} are retweets of {len(rtc)} original tweets")
    print(f"    {sum(rpc.values())} are replies on   {len(rpc)} original tweets")
    print(f"    {sum(qtc.values())} are quotes about {len(qtc)} original tweets")

    return allids

def find_all_with_id(target, twd: dict):
    """
    pass a conversationID or tweetID and return all occurrences in dataset
    :param twd: the full dataset as a dict keyed on tweet ID
    :param target: ID str or list of ID strings
    :return: subset of tweets that matched on any ID field
    """
    retval: dict = {}
    typdef: dict = {'conv': 'conversation', 'id': 'Tweet ID', 'qt_id': 'origin ID for Quote',
                    'rt_id': 'origin ID for ReTweet', 'rply_id': 'origin ID for Reply'
                    }
    def show_matches(typ: str, key: str):
        if typ in typdef:
            print(f"      ID {key} found in dataset as {typdef[typ]}")
        return

    def find_id(x: str):
        retdct: dict = {}
        for k, v in twd.items():
            for idtyp in typdef:
                if idtyp in v and v[idtyp] == x:
                    show_matches(idtyp, k)
                    retdct[k] = v

        return retdct

    print(f"\n    find_all_with_id searches id, conv, qt, rt, and rply ID values")
    if isinstance(target, str):
        print(f"      searching for ID {target}")
        targx: str = target
        retval |= find_id(targx)
    elif isinstance(target, list):
        print(f"      searching target list of {len(target)} IDs")
        for targx in target:
            retval |= find_id(targx)

    return retval
