# coding=utf-8
"""
gs_tweet_analysis is part II of utils for tweets, the first being gs_nlp_util.
gs_tweet_analysis builds word counts, grammar analysis, and other higher level functions
"""
import copy
from math import fabs
import pandas as pd
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gs_data_dictionary import IDIOM_MODS, VADER_MODS

nltk.data.path.append('/Users/bgh/dev/NLTK')
nltk.download('vader_lexicon', download_dir="/Users/bgh/dev/NLTK")
Vsi = SentimentIntensityAnalyzer()
for k, v in VADER_MODS.items():
    Vsi.lexicon[k] = v
for k, v in IDIOM_MODS.items():
    Vsi.constants.SPECIAL_CASE_IDIOMS[k] = v
# updates to Vader constants, I could roll these into a function
# N_SCALAR = -0.74
# B_INCR = 0.293
# B_DECR = -0.293
# C_INCR = 0.733

def filter_dupes(twbatch, dedupe: bool=True, rt_lim: int=0):    # sourcery no-metrics
    """
    use this fx to search duplicates, retweets and quoted tweets in the dataset. this
    function is a key step in assessing the strength of a topical dataset.
      -identifies and removes duplicates by Tweet ID
      -identifies originating Tweet ID for quoted tweets and retweets and counts occurrences.
      -can adjust to limit how many RTs for a single originating ID are kept, as unlike
      quotes or replies, retweets add no new content to a thread.
      -writes a shared record of all Quote text plus metrics by originating tweet ID.

    :param twbatch: list of dict of tweets
    :param dedupe: bool if True removes duplicate Tweet ID's
    :param rt_lim: if 0 don't limit retweets, if >0, allow max number by originating ID
    :return filter_lst: list of tweets with duplicates removed,
        rt_count: RT id's and number of copies in dataset,
        qt_count: QT id's and num of copies in dataset,
        qt_merge: merged Quoted Tweet comments under original Tweet ID
        rply_count: reply ID's and number of replies in dataset for each original ID
    """
    qt_merge: dict = {}
    qt_count: dict = {}
    rply_count: dict = {}
    rt_count: dict = {}
    id_count: dict = {}
    fin_lst: list = []
    dupe_count: int = 0
    dupe_rt: int = 0
    ds_len: int = len(twbatch)

    print(f"\n  -- filter_dupes:\n        starting with {ds_len:,} Tweets       --")
    for twx in twbatch:
        if not isinstance(twx, dict):
            print("    ERROR- filter_dupe2 requires a list of dict as first parameter! ")
            return 13

        tmpdct: dict = twx
        tmpdct['rawtext']: str = copy.copy(twx['text'])
        # identify and remove duplicates by Tweet ID string
        if tmpdct['id'] not in id_count:
            id_count[tmpdct['id']] = 1
        else:
            id_count[tmpdct['id']] += 1
            dupe_count += 1
            if dedupe:
                continue

        if tmpdct.get('qt_id'):
            if isinstance(tmpdct.get('qt_id'), list) and len(tmpdct['qt_id']) == 1:
                idx: str = str(tmpdct['qt_id'][0])
            else:
                idx: str = tmpdct['qt_id']
            if idx not in qt_count:
                # new originating ID found for a quoted tweet
                # 'qt_text' is the text of the originating tweet
                qt_count[idx] = 1
                qt_merge[idx] = {'text': [tmpdct.get('qt_text'), tmpdct.get('text')],
                                 'qrr': tmpdct.get('qt_qrr'), 'fave': tmpdct.get('qt_fave'),
                                 'date': tmpdct.get('qt_src_dt'), 'src_id': [tmpdct.get('id')]}
            else:
                # multiple QT's point to same original
                qt_count[idx] += 1
                qt_merge[idx]['text'].append(tmpdct.get('text'))
                qt_merge[idx]['src_id'].append(tmpdct.get('id'))
                if hasattr(tmpdct, 'qt_qrr') and tmpdct.get('qt_qrr') > qt_merge[idx]['qrr']:
                    qt_merge[idx]['qrr'] = tmpdct.get('qt_qrr')
                if hasattr(tmpdct, 'qt_fave') and tmpdct.get('qt_fave') > qt_merge[idx]['fave']:
                    qt_merge[idx]['fave'] = tmpdct.get('qt_fave')

            if 'qrr' not in qt_merge[idx]:
                qt_merge[idx]['qrr']: int = 0
            elif not isinstance(qt_merge[idx]['qrr'], (int, float)):
                qt_merge[idx]['qrr']: int = 0
            if 'fave' not in qt_merge[idx]:
                qt_merge[idx]['fave']: int = 0
            elif not isinstance(qt_merge[idx]['fave'], (int, float)):
                qt_merge[idx]['fave']: int = 0
            qt_merge[idx]['infl'] = int(qt_merge[idx]['qrr'] + qt_merge[idx]['fave'])

        if tmpdct.get('rply_id'):
            if isinstance(tmpdct.get('rply_id'), list) and len(tmpdct['rply_id']) == 1:
                idx: str = str(tmpdct['rply_id'][0])
            else:
                idx: str = tmpdct['rply_id']
            if idx not in rply_count:
                # new originating ID found for a quoted tweet
                rply_count[idx] = 1
            else:
                # multiple QT's point to same original
                rply_count[idx] += 1

        if tmpdct.get('rt_id'):
            if isinstance(tmpdct.get('rt_id'), list) and len(tmpdct['rt_id']) == 1:
                idx: str = str(tmpdct['rt_id'][0])
            else:
                idx: str = tmpdct['rt_id']
            if idx not in rt_count:
                # first time seeing this retweet, create an rt_count rec for it
                rt_count[idx] = 1
            else:
                # already seen this originating ID, add count, skip if more than 2
                rt_count[idx] += 1
                if 0 < rt_lim < rt_count[idx]:
                    dupe_rt += 1
                    continue

        fin_lst.append(tmpdct)

    print(f"        {dupe_count} duplicate Tweet IDs removed")
    print(f"        {dupe_rt} duplicate RT's removed from dataset")
    print("        -------")
    print(f"        {len(fin_lst):,} records returned --\n")
    if qt_count:
        print("    -- analysis of Quoted Tweets --")
        qt_count = {k: qt_count[k] for k in sorted(qt_count, key=lambda x: qt_count[x], reverse=True)}
        qt_tot: int = sum(qt_count.values())
        print(f"        {qt_tot} quoted tweets for {len(qt_count)} originating Tweets")
        qt_infl: list = [k['infl'] for k in qt_merge.values()]
        qt_infl = sorted(qt_infl, key=lambda x: int(x), reverse=True)
        tot_infl: int = sum(qt_infl[:10])
        print(f"            {tot_infl:,} sum of influence top 10 Quote Tweets")
        qt_max: int = max(qt_count.values())
        print(f"        most QTs in dataset for single originating Tweet: {qt_max}")
        print("        ------------")
    if rply_count:
        print("    -- analysis of Replies in dataset --")
        rply_count = {k: rply_count[k] for k in sorted(rply_count, key=lambda x: rply_count[x], reverse=True)}
        rply_tot: int = sum(rply_count.values())
        print(f"        {rply_tot} replies found for {len(rply_count)} original Tweets")
    if rt_count:
        print("    -- analysis of Retweets in dataset --")
        rt_count = {k: rt_count[k] for k in sorted(rt_count, key=lambda x: rt_count[x], reverse=True)}
        rt_tot: int = sum(rt_count.values())
        print(f"        {rt_tot} RTs found for {len(rt_count)} original Tweets")
        print(f"        retweets limited to {rt_lim} for each originating Tweet ID")
    print("\n  NOTE:  qtmerge contains all quote and original text by originating ID")

    return fin_lst, rt_count, qt_count, qt_merge, rply_count

def find_original_ids(tw_ds: list, rt_ct: dict, qt_ct: dict):
    """
    Fx goes through list of retweet ID's created by filter_rt_and_qt and looks for
    original tweet in dataset.  returns prioritized list of missing original tweets

    :param tw_ds: list of dict, the main Tweet dataset after filter step
    :param rt_ct: dict of RT ids and counts, created in previous step
    :param qt_ct: dict of QT ids and counts, created in previous step
    :return:
    """
    print("\n  FIND_ORIGINAL_IDS seeking original tweet for quotes and retweets")
    print("      %d RTs " % len(rt_ct))
    print("      %d QTs " % len(qt_ct))

    top_select: int = 10
    missing: list = []
    not_missing: list = []
    id_set: set = {x['id'] for x in tw_ds}
    cntr: dict = {"rt_hit": 0, "rt_miss": 0, "qt_hit": 0, "qt_miss": 0}
    rt_ct = {k: rt_ct[k] for k in sorted(rt_ct, key=lambda x: rt_ct[x], reverse=True)}

    def find_tweet(tw_id: str, typ: str = "id"):
        """
        :param tw_id: tweet ID string
        :param typ: default search for ID value, may also be rt_id or qt_id
        :return twx: tweet dictionary record
        """
        return next((twx for twx in tw_ds if twx.get(typ) == tw_id), None)

    print("      searches to perform =   %d" % (len(rt_ct) + len(qt_ct)))
    for findx, findtyp in zip([rt_ct, qt_ct], ["rt", "qt"]):
        for idx in findx:
            if idx in id_set:
                not_missing.append(find_tweet(idx))
                if findtyp in ["rt"]:
                    cntr["rt_hit"] += 1
                elif findtyp in ["qt"]:
                    cntr["qt_hit"] += 1
            else:
                tmp_dct: dict = find_tweet(idx, typ=findtyp + "_id")
                if tmp_dct:
                    tmp_dct["missing_from"] = findtyp
                    missing.append(tmp_dct)
                if findtyp in ["rt"]:
                    cntr["rt_miss"] += 1
                elif findtyp in ["qt"]:
                    cntr["qt_miss"] += 1

    if missing:
        missing: list = sorted(missing, key=lambda x: x.get('qrr') + x.get('fave'), reverse=True)
        qrrfsum: int = 0
        for x, cnt in zip(missing, range(top_select)):
            qrrfsum += int(x['qrr']) + int(x['fave'])
        qrrf_avg = round(qrrfsum / top_select, ndigits=1)
        print("    Originating Tweets found= %d" % len(not_missing))
        print("      from retweets     = %d" % cntr["rt_hit"])
        print("      from quote tweets = %d" % cntr["qt_hit"])
        print("        --------        --------")
        print(f"      missing originals     = {len(missing)}")
        print("               Retweets   %d" % cntr["rt_miss"])
        print("           Quote Tweets   %d" % cntr["qt_miss"])
        print("\n    --  influence of missing Tweets  --")
        print(f"            avg Q-R-R-F top {top_select:,} tweets is {qrrf_avg:,.1f}")
        print("\n    ---- use missing list with 'GET tweets by' ID Twitter API ----")

    return missing, not_missing

def find_rply_quote_ids(tw_ds: list, rply_ct: dict, qt_ct: dict):
    """
    Fx goes through replies and quotes found by filter_dupe2 and looks for
    the original tweet in dataset.  If original is missing, create a list that can
    be used with Twitter get multiple tweets

    :param tw_ds: list of dict, the main Tweet dataset after filter step
    :param rply_ct: dict of reply ids and counts, created in previous step
    :param qt_ct: dict of QT ids and counts, created in previous step
    :return:
    """
    print("\n  find_rply_quote_IDs seeking original tweet for replies and quotes")
    print(f"      {len(rply_ct)} replies ")
    print(f"      {len(qt_ct)} QTs ")
    print(f"      searches to perform =   {(len(rply_ct) + len(qt_ct))}")

    missing: list = []
    not_missing: list = []
    id_set: set = {x['id'] for x in tw_ds}
    cntr: dict = {"rply_hit": 0, "rply_miss": 0, "qt_hit": 0, "qt_miss": 0}
    rply_ct = {k: rply_ct[k] for k in sorted(rply_ct, key=lambda x: rply_ct[x], reverse=True)}
    qt_ct = {k: qt_ct[k] for k in sorted(qt_ct, key=lambda x: qt_ct[x], reverse=True)}

    def find_tweet(tw_id: str, typ: str = "id"):
        """
        :param tw_id: tweet ID string
        :param typ: default search for ID value, can be id, rply_id, rt_id or qt_id
        :return twx: tweet dictionary record
        """
        return next((twx for twx in tw_ds if twx.get(typ) == tw_id), None)

    for findx, findtyp in zip([rply_ct, qt_ct], ["rply", "qt"]):
        for idx in findx:
            if idx in id_set:
                not_missing.append(find_tweet(idx))
                if findtyp in ["rply"]:
                    cntr["rply_hit"] += 1
                elif findtyp in ["qt"]:
                    cntr["qt_hit"] += 1
            else:
                tmp_dct: dict = find_tweet(idx, typ=findtyp + "_id")
                if tmp_dct:
                    tmp_dct["missing_from"] = findtyp
                    missing.append(tmp_dct)
                if findtyp in ["rply"]:
                    cntr["rply_miss"] += 1
                elif findtyp in ["qt"]:
                    cntr["qt_miss"] += 1

    if missing:
        missing: list = sorted(missing, key=lambda x: x.get('qrr') + x.get('fave'), reverse=True)

        print(f"    Originating Tweets found= {len(not_missing)}")
        print(f"    from replies     = {cntr['rply_hit']}")
        print(f"    from quote tweets= {cntr['qt_hit']} ")
        print("    -------------------------")
        print(f"    missing originals       = {len(missing)}")
        print(f"        replies      = {cntr['rply_miss']} ")
        print(f"        quote tweets = {cntr['qt_miss']}   ")
        print("\n    ---- use missing list with 'GET tweets by' ID Twitter API ----")

    return missing, not_missing

def get_word_freq(wrd_list: list, debug: bool = False):
    """
    create dict of distinct words (a 'vocabulary') and number of occurrences
    :param wrd_list: list of tweets, tweet can be list of words, a str, or dict
    :param debug: bool if True will print verbose status
    :return: wordfreq key:str= word, value:int= count of occurrences
    """
    wordfq: dict = {}
    wrd_total: int = 0
    for this_rec in wrd_list:
        if isinstance(this_rec, dict):
            this_seg: str = this_rec['text']
            this_seg: list = this_seg.split()
            for this_w in this_seg:
                if this_w in wordfq:
                    wordfq[this_w] += 1
                else:
                    wordfq[this_w] = 1

        elif isinstance(this_rec, str):
            this_seg: list = this_rec.split()
            for this_w in this_seg:
                wrd_total += 1
                if this_w in wordfq:
                    wordfq[this_w] += 1
                else:
                    wordfq[this_w] = 1

        elif isinstance(this_rec, list):
            for this_w in this_rec:
                wrd_total += 1
                if this_w in wordfq:
                    wordfq[this_w] += 1
                else:
                    wordfq[this_w] = 1

    if debug:
        print("%d unique words from %s total words" % (len(wordfq), "{:,}".format(wrd_total)))
    return wordfq

def count_words(wordlst: list):
    """
    count words in tweet text from list of list, dict, or str
    :param wordlst: list of tweets
    :return: word count, tweet count
    """
    wrd_count: int = 0
    tw_count: int = 0
    for tw in wordlst:
        if isinstance(tw, dict):
            tw_wrds: list = tw['text'].split()
        elif isinstance(tw, str):
            tw_wrds: list = tw.split()
        else:
            tw_wrds: list = tw
        tw_count += 1
        wrd_count += len(tw_wrds)
    return wrd_count, tw_count

def sort_freq(freqdict):
    """
    sort_freq reads word:frequency key:val pairs from dict, and returns a list sorted from
    highest to lowest frequency word
    :param freqdict:
    :return: list named aux
    """
    aux: list = [(v, k) for k, v in freqdict.items()]
    aux.sort(reverse=True)

    return aux

def apply_vader_single(tweetx: dict):
    """
    apply sentiment scoring to a single tweet represented as a dict with tweet fields
    :param tweetx: dict with contents of single tweet
    :return: updated dict with sentiment fields compound, neg, neu, and positive
    """
    if 'text' in tweetx:
        tmpdct: dict = Vsi.polarity_scores(tweetx['text'])
        tweetx |= tmpdct
        print(f" added sentiment score (compound= {tweetx['compound']:.2f} for Tweet {tweetx['id']}")
    return tweetx

def apply_vader(sent_lst: list):
    """
    apply Vader's Valence scoring of words, symbols and phrases for social media sentiment,
    continuous negative-positive range, 4 scores: compound, neg, neutral, and pos.
    application of phrases and idioms, negation and punctuation (ex. ???).

    can add to or modify Vader 'constants' for terms and values.
    Vader is optimized to handle sentiment on short posts like Tweets.
    \n Author Credits:
    Hutto,C.J. & Gilbert,E.E. (2014). VADER: Parsimonious Rule-based Model for Sentiment
    Analysis of Social Media Text. Eighth International Conference on Weblogs and Social
    Media (ICWSM-14). Ann Arbor, MI, June 2014.

    :param sent_lst: list of dict or list of str with Tweet text
    :return: Vscores list of Vader sentiment scores, plus Tweet index info I embedded
    """
    print(f"\n    using Vader to calculate sentiment for {len(sent_lst)} Tweets")
    vscores: list = []
    for snt_x in sent_lst:
        if isinstance(snt_x, list):
            tmpdct: dict = Vsi.polarity_scores(" ".join([str(x) for x in snt_x]))
        elif isinstance(snt_x, str):
            tmpdct: dict = Vsi.polarity_scores(snt_x)
        elif isinstance(snt_x, dict):
            tmpdct: dict = Vsi.polarity_scores(snt_x['text'])
            tmpdct |= snt_x
        else:
            print("apply_vader got incorrectly formatted Tweets as parameter")
            break
        vscores.append(tmpdct)

    cmp_tot: float = 0.0
    neg_tot: float = 0.0
    neu_tot: float = 0.0
    pos_tot: float = 0.0
    v_len = len(vscores)
    for vidx in range(v_len):
        cmp_tot += vscores[vidx]['compound']
        neg_tot += vscores[vidx]['neg']
        neu_tot += vscores[vidx]['neu']
        pos_tot += vscores[vidx]['pos']
    cmp_avg = cmp_tot / v_len
    neg_avg = neg_tot / v_len
    neu_avg = neu_tot / v_len
    pos_avg = pos_tot / v_len
    print(f"Average Vader compound score = {cmp_avg:1.2f} for {v_len} Tweets")
    print(f"Average       negative score = {neg_avg:1.2f} ")
    print(f"Average        neutral score = {neu_avg:1.2f} ")
    print(f"Average       positive score = {pos_avg:1.2f} ")

    return vscores

def apply_vader_quotetweet(sent_lst: list):
    """
    like apply Vader function, but looks for quote tweets and applies to quote tweet
    text.
    Vader is optimized to handle sentiment on short posts like Tweets.
    :param sent_lst: list of dict or list of str with Tweet text
    :return: Vscores list of Vader sentiment scores, plus Tweet index info I embedded
    """
    print(f"\n    RUNNING...using Vader to calculate sentiment for {len(sent_lst)} Tweets")
    vscores: list = []
    added: int = 0
    for twx in sent_lst:
        if not isinstance(twx, dict):
            print("  ERROR- can only apply vader to list of dict of Tweets")
            break
        else:
            if 'qt_text' in twx:
                added += 1
                tmpdct: dict = Vsi.polarity_scores(twx['qt_text'])
                sentfix: dict = {}
                for x in ['compound', 'neg', 'neu', 'pos']:
                    seek: str = f"qt_{x}"
                    sentfix[seek] = tmpdct[x]
                sentfix |= twx
        vscores.append(sentfix)

    cmp_tot: list = [x['qt_compound'] for x in vscores]
    cmp_avg = cmp_tot / len(sent_lst)
    print(f"Average Vader compound score = {cmp_avg:1.2f} for {added} secondary ids")

    return vscores

def apply_vader_qtmerge(sent_lst: list, qtm: dict):
    """
    like apply Vader function, but looks for quote tweets and applies to quote tweet
    text.
    Vader is optimized to handle sentiment on short posts like Tweets.
    :param sent_lst: list of dict or list of str with Tweet text
    :return: Vscores list of Vader sentiment scores, plus Tweet index info I embedded
    """
    print(f"\n    RUNNING...using Vader to calculate sentiment for {len(sent_lst)} Tweets")
    vscores: list = []
    totadd: int = 0
    qtadd: int = 0
    for twx in sent_lst:
        sentfix: dict = {}
        if 'qt_id' in twx:
            if twx['qt_id'] in qtm:
                qtadd += 1
                qttext: str = qtm[twx['qt_id']]['text']
                tmpdct: dict = Vsi.polarity_scores(qttext[len(qttext) - 1])
                for x in ['compound', 'neg', 'neu', 'pos']:
                    seek: str = f"qt_{x}"
                    sentfix[seek] = tmpdct[x]
        totadd += 1
        sentfix |= twx
        vscores.append(sentfix)

    cmp_tot: list = [x['qt_compound'] for x in vscores if 'qt_compound' in x]
    cmp_avg = sum(cmp_tot) / qtadd
    print(f"  Average Vader compound score = {cmp_avg:1.2f} for {qtadd} secondary ids")
    print(f"  total records = {totadd:,} ")

    return vscores

def summarize_vader(vader_scores: list, top_lim: int = 16):
    """
    reports compound, negative, and positive sentiment for tweets in corpus.
    :param vader_scores: list of scores built from apply_vader method
    :param top_lim: integer indicating number of top scores to summarize
    :return: None
    """
    rec_count: int = len(vader_scores)
    print("\nsummarize_vader: Top Sentiment for %d total Tweets:" % rec_count)
    print("    showing top %d scores for compound, neutral, negative and positive sentiment" % top_lim)

    def get_top(scoretyp: str, toplimit: int):
        """
        inner Fx: get top score for score type, sort by descending absolute value
        :param toplimit: number of scores to identify, such as top 10
        :param scoretyp: str to indicate Vader compound, negative, neutral or positive
        :return:
        """
        srtd = sorted(vader_scores, key=lambda x: fabs(x.get(scoretyp)), reverse=True)
        return srtd[:toplimit]

    def describe_score(scoretyp: str):
        """
        gives average, minimum and maximum for a type of sentiment score
        :param scoretyp: str as compound, neu, neg, or pos
        :return: n/a
        """
        typ_tot: float = sum(vader_scores[x][scoretyp] for x in range(rec_count))
        if scoretyp == "neg":
            typestr = "Negative"
        elif scoretyp == "neu":
            typestr = "Neutral"
        elif scoretyp == "pos":
            typestr = "Positive"
        else:
            typestr = "Compound (aggregate)"
        typ_avg: float = typ_tot / rec_count
        typ_min: float = min(vader_scores[x][scoretyp] for x in range(rec_count))
        typ_max: float = max(vader_scores[x][scoretyp] for x in range(rec_count))
        print(f"    {typestr} ", end="")
        print(" Average= %1.3f, Minimum= %1.3f, Maximum= %1.3f \n" % (typ_avg, typ_min, typ_max))

        return

    def show_with_text(typ, tops: list):
        """
        prints applicable sentiment score along with text of Tweet
        :param typ: string with formal Vader sentiment type (neu, pos, neg, compound)
        :param tops: list of top tweets by sentiment type, number of tweets= top_lim
        :return: None
        """
        print("Printing top %d tweets by %s sentiment:" % (top_lim, typ))
        for tws in tops:
            print(f"  {typ} sentiment= {tws[typ]:1.3f} on {tws['tdate']}")
            if typ in ['compound', 'neu', 'pos', 'neg']:
                print(f"{tws['id']}, text: {tws['text'][:110]}")
        return

    for x in ["compound", "neu", "pos", "neg"]:
        describe_score(x)
        top_list = get_top(x, top_lim)
        show_with_text(x, top_list)
        print("")

    return None

def summarize_vader_dict(scored_tws: dict, top_lim: int = 16):
    """
    reports compound, negative, and positive sentiment for tweets in corpus.
    :param scored_tws: a dict of dict
    :param top_lim: integer indicating number of top scores to summarize
    :return: None
    """
    rec_count: int = len(scored_tws)
    print("\nsummarize_vader: Top Sentiment for %d total Tweets:" % rec_count)
    print("    showing top %d scores for compound, neutral, negative and positive sentiment" % top_lim)

    def get_top(scoretyp: str, toplimit: int):
        """
        inner Fx: get top score for score type, sort by descending absolute value
        :param toplimit: number of scores to identify, such as top 10
        :param scoretyp: str to indicate Vader compound, negative, neutral or positive
        :return:
        """
        srtd = sorted(scored_tws, key=lambda x: fabs(scored_tws[x].get(scoretyp)), reverse=True)
        retdct: dict = {k: scored_tws[k] for k in srtd}

        print(f"Printing top {toplimit} tweets for {scoretyp} sentiment:")
        for twkv, ct in zip(retdct.items(), range(toplimit)):
            if scoretyp in twkv[1] and 'tdate' in twkv[1]:
                print(f"  {scoretyp} sentiment= {twkv[1][scoretyp]:1.3f} on {twkv[1]['tdate']}")
                print(f"{twkv[0]}, text: {twkv[1]['text']}")
            else:
                print(f" could not find {scoretyp} in Tweet {twkv[0]}")
        return

    def describe_score(scoretyp: str):
        """
        gives average, minimum and maximum for a type of sentiment score
        :param scoretyp: str as compound, neu, neg, or pos
        :return: n/a
        """
        typ_tot: float = sum([x[scoretyp] for x in scored_tws.values()])
        if scoretyp == "neg":
            typestr = "Negative"
        elif scoretyp == "neu":
            typestr = "Neutral"
        elif scoretyp == "pos":
            typestr = "Positive"
        else:
            typestr = "Compound (aggregate)"
        typ_avg: float = typ_tot / rec_count
        typ_min: float = min([x[scoretyp] for x in scored_tws.values()])
        typ_max: float = max([x[scoretyp] for x in scored_tws.values()])
        print(f"    {typestr} ", end="")
        print(" Average= %1.3f, Minimum= %1.3f, Maximum= %1.3f \n" % (typ_avg, typ_min, typ_max))

        return

    for x in ["compound", "neu", "pos", "neg"]:
        describe_score(x)
        get_top(x, top_lim)
        print("")

    return None

def get_next_by_val(lst, field: str, val: float):
    """
    takes a list of dict, sorts by descending value of chosen field, then finds first matching
    index value (ordinal ID number) which is LESS THAN the identified target value
    :param lst: a list of dict, that is: list of Tweets where dict keys are Tweet fields
    :param field: str field name for retweet/quote count, fave count or sentiment value
    :param val: integer or float value to be found in sorted field values
    :return: ordinal index number, or -1 if error/not found
    """
    lenx: int = len(lst)
    if field in {'compound', 'neg', 'pos'}:
        srtd: list = sorted(lst, key=lambda x: fabs(x.get(field)), reverse=True)
    else:
        srtd: list = sorted(lst, key=lambda x: x.get(field), reverse=True)

    for x in range(lenx):
        if field in {'compound', 'neg', 'pos'}:
            if fabs(srtd[x][field]) <= fabs(val):
                return x
        elif int(srtd[x][field]) <= val:
            return x
    return -1

def get_pctle_sentiment(twlst: list, ptile: int = 0, quota: int = 0):
    """
    create list of Tweets in top percentile for compound sentiment score
    :param twlst: list of Vader scores
    :param ptile: integer from 0 to 99 indicating percentile above which to include
    :param quota: alternate to percentile is to specify quota-  x records to select
    :return: list of str: Tweets in top quartile by sentiment
    """
    totlen: int = len(twlst)
    srtd: list = sorted(twlst, key=lambda x: fabs(x.get('compound')), reverse=True)
    if quota != 0:
        print("selecting top %d tweets by quota provided" % quota)
    else:
        top_pcnt: int = 100 - ptile
        quota = int(round(totlen * (top_pcnt / 100), ndigits=0))

    print("\n get_pctle_sentiment: selecting top %d Tweets out of %d" % (quota, totlen))

    tops: list = srtd[:quota]
    med_sent: float = tops[round(quota * 0.5)]['compound']
    top_sent: float = tops[0]['compound']
    sent_80: int = get_next_by_val(twlst, "compound", 0.80)
    print("    compound sentiment of 0.8 occurs at rec %d of %d" % (sent_80, totlen))
    print("    filtered: top sentiment is %1.2f, median is %1.2f" % (top_sent, med_sent))
    print("      least (abs) sentiment in filtered is: %1.3f" % tops[quota - 1]['compound'])

    return tops

def get_neg_sentiment(twlst: list, cutoff: float = 0.2):
    """
    create list of Tweets in top quartile for negative sentiment score
    :param twlst: list of Vader scores
    :param cutoff: minimum score to include
    :return: list of str: Tweets in top quartile by sentiment
    """
    totlen: int = len(twlst)
    sent_4: int = get_next_by_val(twlst, "neg", cutoff)
    srtd: list = sorted(twlst, key=lambda x: x.get('neg'), reverse=True)
    tops: list = srtd[:sent_4]
    print("\n get_negative_sentiment: selecting %d Tweets out of %d" % (sent_4, totlen))

    med_sent: float = tops[round(sent_4 * 0.5)]['neg']
    top_sent: float = tops[0]['neg']
    print("    filtered: top sentiment is %1.2f, median is %1.2f" % (top_sent, med_sent))

    return tops

def get_pctle_qrr(twlst: list, ptile: int = 0, quota: int = 0):
    """
    create list of Tweets in top quartile for qrr count
    :param twlst: list of dict of Tweets w/quoted/retweeted/reply counts
    :param ptile: integer from 0 to 99 indicating percentile above which to include
    :param quota: identify an integer number of records instead of a percentile
    :return: list of dict: Tweets in top quartile by popularity count
    """
    totlen: int = len(twlst)
    srtd: list = sorted(twlst, key=lambda x: x.get('qrr'), reverse=True)
    if quota != 0:
        print("selecting top %d tweets by quota provided" % quota)
    else:
        top_pcnt: int = 100 - ptile
        quota = int(round(totlen * (top_pcnt / 100), ndigits=0))

    print("\n get_pctle_qrr: getting top %d Tweets out of %d by qrr count" % (quota, totlen))

    tops: list = srtd[:quota]
    midpoint: int = int(round(quota * 0.5, ndigits=0))
    med_qrr: int = tops[midpoint]['qrr']
    top_qrr: int = tops[0]['qrr']
    qrr_80: int = get_next_by_val(twlst, "qrr", 80)
    print("    qrr of 100 occurs at record %d of %d" % (qrr_80, totlen))
    print("    filtered: top qrr is %d, median is %d" % (top_qrr, med_qrr))
    print("      least included qrr is: %d" % (tops[quota - 1]['qrr']))

    return tops

def get_pctle_fave(twlst: list, ptile: int = 0, quota: int = 0):
    """
    create list of Tweets in top quartile for qrr count
    :param twlst: list of dict of Tweets w/ favorite counts
    :param ptile: integer from 0 to 99 indicating percentile above which to include
    :param quota: alternate to percentile is to specify quota-  x records to select
    :return: list of dict: Tweets in top quartile by popularity count
    """
    totlen: int = len(twlst)
    srtd: list = sorted(twlst, key=lambda x: x.get('fave'), reverse=True)
    if quota != 0:
        print("selecting top %d tweets by quota provided" % quota)
    else:
        top_pcnt: int = 100 - ptile
        quota = int(round(totlen * (top_pcnt / 100), ndigits=0))

    print("\n get_pctle_fave: getting top %d Tweets of %d by fave count" % (quota, totlen))

    tops: list = srtd[:quota]
    midpoint: int = int(round(quota * 0.5, ndigits=0))
    med_fave: int = tops[midpoint]['fave']
    top_fave: int = tops[0]['fave']
    fave_10: int = get_next_by_val(twlst, "fave", 10)
    print("    fave count of 10 is at record %d out of %d" % (fave_10, totlen))
    print("    filtered: top fave is %d, median is %d" % (top_fave, med_fave))
    print("      least fave included is: %d" % tops[quota - 1]['fave'])

    return tops

def clean_hashtags(hsh: dict, h_stp: list=None, min_f: int=5):
    """
    utility to apply stop words for hashtags and user mentions,
    :param hsh: dict key=hashtag, value=number of occurrences
    :param h_stp: list of STOPWORDS to remove from hashtag list
    :param min_f: minimum frequency for words to be included, default=5
    :return sorted dict (by descending value) with stops removed
    """
    h2 = sorted(hsh, key=lambda x: hsh.get(x), reverse=True)
    if h_stp:
        hcln: dict = {k: hsh[k] for k in h2 if k not in h_stp}
    else:
        hcln: dict = {k: hsh[k] for k in h2}

    return {k: v for k, v in hcln.items() if v > min_f}

def split_toplist_bytyp(df: pd.DataFrame):
    """
    returns three df's after splitting into 'type' groups:
        1. top percentile sentiment (type 'qfs', 'qs' and 'fs)
        2. both influence metrics (type 'qf')
        3. just 'q' or just 'f'
    Q = high Quote-Retweet-Reply count, F= high favorited-liked count, and
    S= high sentiment score (in absolute sense- high pos or high neg, -1.0 to +1.0)
    :param df: tweet dataframe
    :return:
    """
    s_df: pd.DataFrame = df.loc[df['type'].isin(['qfs', 'qs', 'fs']), ]
    qftyp_df: pd.DataFrame = df.loc[df['type'].isin(['qf']), ]
    qorf_df: pd.DataFrame = df.loc[df['type'].isin(['q', 'f']), ]

    return s_df, qftyp_df, qorf_df

def get_delta_if_qt(twd: dict, tdx: set):
    """
    for a tweet, calc the difference between compound sentiment of originating tweet and
    the quote or reply if this is a quote or reply (if not returns None).
    in the case of a reply, searches for original in the dataset,
    for quote tweets, both sentiment scores should already exist as field 'qt_compound'.
    :param twd: a dictionary for a single tweet
    :param tdx: dict of dict of tweet's keyed on ID
    :return: updated dictionary with 'sent_delta' field
    """
    no_reply: dict = {}
    # deltas can be computed from qt record, or by looking up the originating tweet
    if 'qt_compound' in twd:
        twd['sent_delta'] = twd['compound'] - twd['qt_compound']
    if 'qt_qrr' in twd:
        twd['qrr_delta'] = twd['qrr'] - twd['qt_qrr']
    if 'qt_fave' in twd:
        twd['fave_delta'] = twd['fave'] - twd['qt_fave']

    if 'rply_id' in twd:
        if twd['rply_id'] in tdx:
            twd['sent_delta'] = twd['compound'] - tdx[twd['rply_id']]['compound']
        elif twd['rply_id'] in no_reply:
            no_reply[twd['rply_id']] += 1
        else:
            no_reply[twd['rply_id']] = 1

    return twd, no_reply
