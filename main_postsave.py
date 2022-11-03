"""
import, clean, and filter tweets about 'work', 'job', and sometimes 'quit', or 'suck'.
this app reuses pieces I built for Gamestop and Superleague analyses.
gs_data_dictionary.py holds constants that can be tweeked for a particular twitter/nlp analysis
"""

import re
import pandas as pd
import datetime as dt
from statistics import quantiles
from gs_data_dictionary import MODELDIR, OUTDIR, GS_STOP, STOP_TWEET, STOP_CLOUD, TWEET_RECORD, NOT_ALPHA
import nlp_util as util
import filter_and_enrich as gsfe
import gs_nlp as gsnlp
import tweet_visualize as gsviz
import gs_pandas as gstp
import coverage_patterns as gscp
module_path = MODELDIR

# this script #2 starts by reading file saved by gsutil.save_dataset in main script
load_tweet_dataset: bool = True
load_pandas: bool = True
hashes_mentions: bool = True
topic_modeling: bool = True
run_scaling: bool = True
show_tweet_calcs: bool = True
prep_labor_Data: bool = True
show_plots: bool = True
run_tfidf: bool = True
run_cloud: bool = True
check_coverage: bool = False
trace_analysis: bool = False
word_valuation: bool = False

CROP_START: str = "2022-05-01 06:00"
CROP_END: str = "2022-10-20 18:00"
STARTD: str = "2022-05-01"
ENDD: str = "2022-10-20"
ttopic: str = "Nature of Work"
projname: str = "Workplace Dynamics"
hash_stops: list = ['quietquitting', 'inhomecare', 'caregiver', 'seniorcare', 'caregiverjob',
                    'caregivingjob', 'applytoday', 'alwayshiring', 'localjobs', 'nowhiring']

tw_dataset_file: str = "WorkplaceDynamics_14469_2022-11-01.json"
hashes_file: str = "Work_hashtags_4804.json"
tokens_file: str = "Workplace_wordtokens_14469.json"
quotetweet_f: str = "quotetweets_1342.json"
time_use_f: str = "bls_us_2016-2021_timeuse.csv"
productive_f: str = "bls_labor_2012-2022.csv"

date_col: str = "sent"
log_cols: list = ['qrr', 'fave', 'infl', 'quot', 'rtwt', 'rply']
scale_cols: list = ['qrr', 'fave', 'infl', 'pos', 'neg', 'neu', 'compound']

if load_tweet_dataset:
    util.box_prn("Loading saved Tweet dataset files- XXX".replace("XXX", ttopic))
    tw_full: list = util.load_tweet_archive(archive_f=tw_dataset_file, workdir=OUTDIR, twrec=TWEET_RECORD)
    word_tokens = util.load_dict(fname=tokens_file, desc="token lists")
    cleanread = util.convert_datetime_to_str(tw_full)
    tw_clean = util.filter_by_dates(cleanread, startd=STARTD, endd=ENDD, dnam='tdate')
    corpustext = util.flatten_list_dict(tw_clean, addq=True)
    # qt_merge = gsutil.load_dict(fname=quotetweet_f, desc="quoted tweets")

if load_pandas:
    # basic_df: pd.DataFrame = pd.DataFrame.from_records(tw_full)
    all_df = gstp.create_dataframe(tw_clean, dcol="sent")
    all_df: pd.DataFrame = gstp.crop_df_to_date(all_df, CROP_START, CROP_END)
    scale_df = gstp.prep_scored_tweets(all_df, logc=log_cols, zc=scale_cols)
    # alltw_dct = alltw_df.to_dict("records")

if hashes_mentions:
    if hash_list := gsnlp.get_hashtags(tw_clean):
        hash_cln: dict = gsfe.clean_hashtags(hash_list, ["quietquitting"])

if topic_modeling:
    corp_clean = gsnlp.prep_for_lda(corpustext)
    docBow, docDict = gsnlp.gensim_doc_terms(corp_clean)
    ntopics: int = 10
    lda_mdl = gsnlp.run_lda_model(doc_term=docBow, term_dict=docDict, word_topics=True, topics=ntopics)
    gsnlp.display_lda(lda_mdl, ntopic=ntopics)
    # gsnlp.get_top_terms(lda=lda_mdl, dDict=docDict, tpcs=8, tpc_trms=6)
    # gsnlp.get_coherence(lda_mdl, docDict, corp_clean)
    # gsnlp.save_lda_model(lda_mdl)

if run_scaling:
    def add_redist(twl):
        tmplst: list = []
        for tw in twl:
            if isinstance(tw['sent'], (dt.datetime, pd.DatetimeTZDtype)):
                tw['tdate']: str = dt.datetime.strftime(tw['sent'], "%Y-%m-%d %H:%M")
            # create 'influence' column as sum of quote, retweet, reply, and Liked counts
            tw['infl'] = tw['qrr'] + tw['fave']
            tw['redist'] = 1
            if 'rply_id' in tw and len(tw['rply_id']) > 3:
                tw['redist'] += 1
            elif 'rt_id' in tw and len(tw['rt_id']) > 3:
                tw['redist'] += 1
            elif 'qt_id' in tw and len(tw['qt_id']) > 3:
                tw['redist'] += 1
            tmplst.append(tw)
        return tmplst
    # general function to apply discrete quartile code for a metric, useful for visualizations
    def quart_codes(x, quarts):
        if x < quarts[0]:
            return 1
        elif x < quarts[1]:
            return 2
        elif x < quarts[2]:
            return 3
        else:
            return 4

    top70pct = gsnlp.final_toplist(twlst=tw_clean, topcut=0.7)
    quartcomp = quantiles([x['compound'] for x in tw_clean])
    for x in tw_clean:
        x['compcode'] = quart_codes(x['compound'], quarts=quartcomp)

    top70pct = add_redist(top70pct)

if show_tweet_calcs:
    # tw_bytag = gsnlp.getby_textortag(tw_clean, twtxt=['lifework'], twtag=['quietquitting', 'worklifebalance'])

    print(f" top Quote-RT-Reply count for single tweet: {max(x['qrr'] for x in top70pct):,}")
    print(f" top Like count for single tweet: {max(x['fave'] for x in top70pct):,}")
    quartneu = quantiles([x['neu'] for x in top70pct])
    quartneuall = quantiles([x['neu'] for x in cleanread])
    quartneg70 = quantiles([x['neg'] for x in top70pct])
    quartnegall = quantiles([x['neg'] for x in cleanread])
    print(f"\n neutral sentiment quartiles, top 70percentile influence: {quartneu}")
    print(f" neutral sentiment quartiles, all tweets: {quartneuall}")
    print(f"\n negative sentiment quartiles, top 70percentile influence: {quartneg70}")
    print(f" negative sentiment quartiles, all tweets: {quartnegall}")

if prep_labor_Data:
    plt_lay = gsviz.create_layout()

    pct70_df = gstp.create_dataframe(twlist=top70pct, dcol="tdate")
    scale70df = gstp.prep_scored_tweets(pct70_df, logc=log_cols, zc=scale_cols)
    print("\n  Distribution of attribute Z-scores, top 30 percentilen tweets by infl")
    print(scale70df[[x for x in scale70df.columns if str(x).endswith("zsc")]].describe())
    print("\n  Distribution of attributes with robust scaling, top 30 percentilen tweets by infl")
    print(scale70df[[x for x in scale70df.columns if str(x).endswith("rsc")]].describe())
    print("\n  Distribution of log-based attributes")
    print(scale70df[[x for x in scale70df.columns if str(x).endswith("log")]].describe())

    atu_df: pd.DataFrame = gsviz.get_employee_timeuse(time_use_f)
    atu_df = atu_df.drop(columns=['Label'])
    bls_df: pd.DataFrame = gsviz.get_bls_data(productive_f)
    bls_df: pd.DataFrame = bls_df[bls_df.Units == "Indexed to 2012"]
    blsdf: pd.DataFrame = bls_df[bls_df.Sector == "Business-nonfarm workers"]
    blsdf: pd.DataFrame = blsdf.drop(columns=['Sector', 'Units'])
    pltdct = blsdf.to_dict("records")

if show_plots:
    figsnt_hist = gsviz.show_sent_distribution(scale70df, plt_lay, appd=projname)
    figsnt_hisall = gsviz.show_sent_distribution(scale_df, plt_lay, appd=projname)

    fig_histqrr = gsviz.hist_quot_rply_rtwt(scale_df, plyt=plt_lay, appd=projname)
    figqrr_hist = gsviz.hist_quot_rply_rtwt(scale70df, plyt=plt_lay, appd=projname)

    fig_hash = gsviz.bar_hashtags(hashes=hash_cln, stops=hash_stops, appd=projname)
    fig_horizh = gsviz.bar_tags_horizontal(hash_cln, plyt=plt_lay, stops=hash_stops, appd=projname)
    # plotdf: pd.DataFrame = gstp.set_dist_sent_coding(pct70_df)
    fig_3d = gsviz.plot3d_bydate(scale70df, plt_lay, appd=projname)
    f3dlst = gsviz.plot_3d_from_list(top70pct, plyt=plt_lay, appd=projname)

    fig_labscat = gsviz.labor_scatter(blsdf, plyt=plt_lay, appd=projname)

    # a little housekeeping prior to showing time use data:
    atu_idxdict: dict = {"Employed Work": 1, "Watching-Streaming": 2, "Parenting-resident children u18": 3,
                         "Socializing-Communicating": 4, "Housework": 5}
    atu_df['ttyp'] = atu_df.Series.apply(lambda x: atu_idxdict[x] if atu_idxdict.get(x) else 6)
    atu_df['yr'] = atu_df['Year'].copy()
    atu_df.set_index(['ttyp', 'yr'], drop=True, inplace=True, append=False)
    atu_df = atu_df.sort_index()
    atu_df.reset_index()
    gsviz.plot_atudata(atu_df, plyt=plt_lay, appd=projname)

    fig_p3d= gsviz.plot_3d(scale70df, plyt=plt_lay, appd=projname)

if run_tfidf:
    util.box_prn("TF*IDF using oneRT corpus...determining importance of words")
    # FIRST, clear any additional stop words found in reviewing word tokens
    corpus2: list = []
    for twx in corpustext:
        if clntx := str(twx).encode('utf-8').decode('ascii', 'ignore'):
            alpha_only: str = re.sub(NOT_ALPHA, " ", clntx)
            alpha_only: str = str(re.sub("(\s{2,})", " ", alpha_only)).strip()
            corpus2.append(alpha_only)
    corpus2: list = util.do_stops(corpus2, stop1=GS_STOP, stop2=STOP_TWEET)
    tfi_wrds = util.do_wrd_tok(corpus2)
    wrd_freq = gsnlp.calc_tf(corpus2, word_tokens=True, calctyp="UNIQ")
    tws_wrd = gsnlp.count_tweets_for_word(wrd_freq)
    tw_idf = gsnlp.calc_idf(wrd_freq, tws_wrd)
    tf_idf = gsnlp.calc_tf_idf(wrd_freq, tw_idf)
    tfi_avg: dict[str, float] = gsnlp.calc_corpus_tfidf(tf_idf, calctyp="AVG")
    tfi_avg = {k: tfi_avg[k] for k in sorted(tfi_avg, key=lambda x: tfi_avg.get(x), reverse=True)}

    TFI_STOP = gsnlp.do_tfidf_stops(tfi_avg)
    util.box_prn(f"tfidf added {len(TFI_STOP)} stop words")

    # this next step is messed up, needs debug
    tw_stop_tfidf = util.do_stops(tfi_wrds, stop1=TFI_STOP)

if run_cloud:
    # handy to see what we have after pre-processing and filtering
    adhoc_stops: list = ['sultan', 'shirley', 'gladys', 'richard', 'naman',
                         'portuguese', 'wayne', 'playoff', 'allen', 'fluid',
                         'transmission', 'stan', 'lung', 'chile', 'putin',
                         'wheels', 'emily', 'bengaluru', 'lungs', 'wyoming',
                         'shoe', 'slice', 'lsas', 'more', 'help', 'read',
                         'mean', 'getting', 'what', 'when', 'someone', 'give',
                         'over', 'come', 'were', 'dont', 'does', 'always',
                         'after', 'getting', 'show', 'like', 'today', 'only',
                         'know', 'want', 'hour', 'every', 'where', 'have',
                         'been', 'same']
    wrd_cld = util.do_stops(corpus2, stop1=STOP_CLOUD)
    wrd_clda = gsnlp.cleanup_for_cloud(wrd_cld)
    gsviz.do_cloud(wrd_clda, opt_stops=adhoc_stops)

    adhoc_stop2: list = ['when', 'where', 'after', 'give', 'always', 'only', 'before',
                         'over', 'wanna', 'want', 'been', 'good', 'make', 'does', 'were',
                         'does', 'would', 'thing', 'less', 'fuck', 'hour', 'yeah',
                         'aint', 'whats', 'wont', 'gotta', 'youll', 'blow', 'fucking',
                         'gonna', 'india', 'nice', 'girl', 'yall', 'cool', 'sorry',
                         'tried', 'period', 'arent', 'probably', 'couple', 'shut',
                         'happens']
    # alternative cloud without tfidf stop removal
    wrd_cld2 = util.do_stops(tw_stop_tfidf, stop1=STOP_CLOUD)
    wrd_cld2b = gsnlp.cleanup_for_cloud(wrd_cld2)
    gsviz.do_cloud(wrd_cld2b, opt_stops=adhoc_stop2, minlen=4, maxwrd=80)

if check_coverage:
    # make sure there are sentiment scores for QT text in dataset:
    # twfull2 = gsta.apply_vader_qtmerge(tw_clean, qt_merge)
    fulldict: dict = {x['id']: x for x in tw_clean}
    twids: set = set(fulldict)
    rp_dct: dict = {}
    for x in fulldict.values():
        if 'rply_id' in x:
            if x['rply_id'] in rp_dct:
                rp_dct[x['rply_id']] += 1
            else:
                rp_dct[x['rply_id']] = 1
    rpids: set = set(rp_dct)
    qt_dct: dict = {}
    for x in fulldict.values():
        if 'qt_id' in x:
            if x['qt_id'] in qt_dct:
                qt_dct[x['qt_id']] += 1
            else:
                qt_dct[x['qt_id']] = 1
    qtids: set = set(qt_dct)
    rt_dct: dict = {}
    for x in fulldict.values():
        if 'rt_id' in x:
            if x['rt_id'] in rt_dct:
                rt_dct[x['rt_id']] += 1
            else:
                rt_dct[x['rt_id']] = 1
    rtids: set = set(rt_dct)
    id_dct: dict = {}
    for x in fulldict.values():
        if 'rt_id' in x:
            if x['rt_id'] in id_dct:
                id_dct[x['rt_id']] += 1
            else:
                id_dct[x['rt_id']] = 1
    ids: set = set(id_dct)

    # conv_nodupe = gstt.identify_conversations(twd=fulldict, qtid=qtids, rpid=rpids, rtid=rtids)
    check_cvrg = gscp.check_tweet_activity_coverage(fulldict, twdc=id_dct, rpdc=rp_dct, qtdc=qt_dct, rtdc=rt_dct)
    rpqt_cvrg = gscp.check_rpqt_coverage(fulldict, twdc=id_dct, rpdc=rp_dct, qtdc=qt_dct, rtdc=rt_dct)

    check_cvrg = {k: check_cvrg[k] for k in sorted(check_cvrg, key=lambda x: check_cvrg[x].get('online_less_ds') if
    check_cvrg[x].get('online_less_ds') else 0, reverse=True)}

    rpqt_cvrg = {k: rpqt_cvrg[k] for k in sorted(rpqt_cvrg, key=lambda x: rpqt_cvrg[x].get('online_less_ds') if
    rpqt_cvrg[x].get('online_less_ds') else 0, reverse=True)}

    rpqt_cvrg2 = gscp.update_scnd_from_primary(fulldict, rpqt_cvrg)
    coverage_final = gscp.get_best_metric_values(rpqt_cvrg2)
    coverage_final = {k: coverage_final[k] for k in sorted(coverage_final, key=lambda x: coverage_final[x].get(
        'online_less_ds') or 0, reverse=True)}

    priority_ids: list = []
    for cvg, ct in zip(coverage_final.items(), range(3000)):
        priority_ids.append(cvg[0])
    top_miss, top_have = gscp.split_influential_into_missing_ornot(priority_ids, twdx=twids)
    # that was the key step- now we know the top priority Tweets to GET,
    # and we know the most influential tweets in our dataset for plotting and analyzing!
    util.print_missing_for_postman(msng=top_miss, outname="top_vip_missing")

    vip_tweets: dict = {}
    verify_ct: int = 0
    for x in top_have:
        if x in twids:
            vip_tweets[x] = fulldict[x]
            verify_ct += 1
    print(f"\n    verified VIP tweets in dataset: {verify_ct}")

    figsnt_hist = gsviz.show_sent_fromlist(vip_tweets, plt_lay, appd=projname)

if trace_analysis:
    # as RT's add no new content, let's leave them out of the union...
    scndry_ids: set = rtids.union(rpids.union(qtids))
    scndry_only: set = scndry_ids.difference(twids)

    twtrace = gscp.build_linked_trace(fulldict)
    twtrace: dict = {k: twtrace[k] for k in sorted(twtrace, key=lambda x: len(twtrace[x]), reverse=True)}
    tracelength =gscp.get_trace_depth(twtrace)
    miss_trace = gscp.get_keys_missing_metrics(twtrace, lvl0=True)

    scnd_fill = gscp.multi_enrich_second(fulldict, id2nd=scndry_only)
    second_qt = gscp.enrich_secondary_ids(fulldict, id2nd=scndry_only, typ='qt')
    second_rp = gscp.enrich_secondary_ids(fulldict, id2nd=scndry_only, typ='rply')
    second_rt = gscp.enrich_secondary_ids(fulldict, id2nd=scndry_only, typ='rt')

    trace2 = gscp.fill_trace_fromgets(second_qt, twtrace)
    trace3 = gscp.fill_trace_fromgets(second_rp, trace2)
    trace4 = gscp.fill_trace_fromgets(second_rt, trace3)

    gscp.count_missing_types(twtrace)
    trace2 = gscp.fill_trace_fromgets(second_qt, twtrace)

if word_valuation:
    # split words into 3 categories by tfidf values, then score tweets by
    # adding 2 for each great word, 1 for a mid-level word,
    # and -1 for every 3 weak words used
    tfi_avg = {k: tfi_avg[k] for k in sorted(tfi_avg, key=lambda x: tfi_avg.get(x), reverse=True)}
    tiles = quantiles(tfi_avg.values(), n=3)
    TFI_GREAT: list = []
    TFI_GOOD: list = []
    TFI_BAD: list = []
    for k, v in tfi_avg.items():
        if v >= tiles[1]:
            TFI_GREAT.append(k)
        elif v >= tiles[0]:
            TFI_GOOD.append(k)
        else:
            TFI_BAD.append(k)

    TFI_GREAT = set(TFI_GREAT)
    TFI_GOOD = set(TFI_GOOD)
    TFI_BAD = set(TFI_BAD)

    tw_scored = gsnlp.score_good_words(cleanread, TFI_GREAT, TFI_GOOD, TFI_BAD)

    # ad-hoc viewing of key df columns for further filtering:
    # shows date, id, text, hashes, word score and sentiment
    # scale_df.iloc[:, [7, 9, 5, 19, 22, 1, 3]]
    best_tweets: list = ['1547987670785134592', ]
