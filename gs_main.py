"""
import, clean, and filter and analyze tweets
gs_data_dictionary.py holds constants that can be tweeked for a particular twitter/nlp analysis
"""
import timeit
import pandas as pd
import re
from gs_data_dictionary import TW_DIR, RAWDIR, GS_STOP, STOP_TWEET, NOT_ALPHA, STOP_NONALPHA, \
    TOPIC_WORDS, GOOD_HASHES, ANTI_TOPICS, BAD_HASHES, BAD_IDS
import nlp_util as util
import filter_and_enrich as gsfe
import gs_pandas as gsp
import gs_nlp as gsn
import tweet_gensim as twgs

projname: str = "Workplace Dynamics"
CROP_START: str = "2022-05-01 06:00"
CROP_END: str = "2022-10-22 18:00"
start_dt = '2022-05-01'
end_dt = '2022-10-22'
date_col: str = "sent"
# SCRIPT CONTROL: set booleans to control which script section are run below
clean_raw_dataset: bool = True
do_culling: bool = True
check_dates: bool = True
build_dataframes: bool = False
run_sentiment: bool = True
run_word_token: bool = True
run_tfidf: bool = True
run_word_scoring: bool = False
save_dataset: bool = False

if inputfiles := util.get_file_list(TW_DIR):
    tw_raw: list = util.read_json_files(inputfiles, TW_DIR)
    tweets_pre = util.get_fields_simple(tw_raw)

if clean_raw_dataset:
    # filter out duplicates and off-topic tweets and identify missing tweets
    util.box_prn([f"parsed {len(tweets_pre)} raw Tweets,", "starting filtering and scrubbing"])

    if userfiles := util.get_file_list(jsondir=RAWDIR):
        users_raw: list = util.read_json_files(userfiles, RAWDIR)
        users = util.parse_user_data(users_raw)

    tw_rtfilt, rt_count, qt_count, qt_merge, rply_count = gsfe.filter_dupes(tweets_pre, rt_lim=3, dedupe=True)
    missing, not_missing = gsfe.find_rply_quote_ids(tw_rtfilt, rply_ct=rply_count, qt_ct=qt_count)
    misslst, notmissed = util.filter_tags_words(missing, topicw=TOPIC_WORDS,
                                                offtop=ANTI_TOPICS, gdhsh=GOOD_HASHES)
    util.print_missing_for_postman(msng=misslst, outname="tweets_missing8")
    hash_list = gsn.get_hashtags(tw_rtfilt)

    tw_scrub: list = util.scrub_tweets(tw_rtfilt, strict=False)
    tw_ascii = util.strip_nonascii_tweets(tw_scrub)
    tw_stops = util.remove_parens_stops(tw_ascii, stop1=GS_STOP, stop2=STOP_TWEET)
    tw_postparse = util.make_tags_lowercase(tw_stops)

    tw_alpha = util.remove_parens_stops(tw_stops, stop1=STOP_NONALPHA)

if do_culling:
    ds1: list = []
    if BAD_IDS:
        for x in tw_postparse:
            if x['id'] not in BAD_IDS:
                ds1.append(x)
    print("  removed known off-topic ID's, resulting set has {len(ds1)} tweets")

    ds2 = util.filter_antitags(ds1, antiwrds=ANTI_TOPICS, antihsh=BAD_HASHES)
    ds3, tw_offtopic= util.filter_tags_words(ds2, topicw=TOPIC_WORDS,
                                                    offtop=ANTI_TOPICS, gdhsh=GOOD_HASHES)
    ds4 = util.cull_low_quality(ds3, maxrepeat=5)
    tw_clean, miss_users = util.cleanup_tweet_records(ds4, users, chktyps=True)

    util.print_missing_for_postman(msng=miss_users, outname="users_missing")
    tweets_final = util.filter_ds_start_end(tw_clean, startd=start_dt, endd=end_dt)

if check_dates:
    tw_crop = util.get_dates_in_batch(tweets_final)
    tws_day = [sum(ov.values()) for ov in tw_crop.values()]
    totdays = len(tws_day)
    max_day = max(tws_day)
    line_2: str = f"days with tweets: {totdays}, max tweets single day: {max_day}"
    util.box_prn(["distribution of Tweets by hour and day", line_2])
    # gsutil.print_day_hour_distro(tw_days)
    # gsutil.print_distro_bydate(tw_crop)
    # tws_light = gsutil.keep_text_date_metrics_only(tweets_final, qt_merge)

if build_dataframes:
    # create pandas Dataframe for all Tweets and crop to start and end dates
    tweetdf: pd.DataFrame = gsp.create_dataframe(tweets_final)
    tweetdf = tweetdf.sort_values("sent")
    tweetdf.reset_index(inplace=True, drop=True)
    tweetdf: pd.DataFrame = gsp.crop_df_to_date(tweetdf, CROP_START, CROP_END)
    tw_final: list = tweetdf.to_dict("records")

if run_sentiment:
    # Uses nltk-Vader for sentiment. summarize_vader displays sentiment by type
    # filter tweets for topic words prior to scoring for more focused results!
    util.box_prn(f"Sentiment Scoring for {len(tweets_final):,} on {projname}")
    tw_post_pos = gsn.check_tweet_POS(tweets_final)
    tw_sentiment = gsfe.apply_vader(tw_post_pos)
    tw_sentplus = gsfe.apply_vader_qtmerge(tw_sentiment, qtm=qt_merge)
    # gsfe.summarize_vader(tw_sentplus)

if run_word_token:
    util.box_prn("running WORD TOKENIZATION section")
    tweet_scrubd: dict = util.iter_scrub_todict(tw_sentplus, strict=True)
    tw_flatdict = util.flatten_dict_dict(tweet_scrubd, addq=True)
    # do_wrd_tok needs debugging
    # words_clean = gsutil.do_wrd_tok(tw_flatdict)
    corpus_dct: dict = {}
    for k, v in tw_flatdict.items():
        alpha_only: str = re.sub(NOT_ALPHA, " ", v)
        alpha_only: str = str(re.sub("(\s{2,})", " ", alpha_only)).strip()
        corpus_dct[k] = alpha_only

    corpus2: list = util.do_stops(corpus_dct.values(), stop1=GS_STOP, stop2=STOP_TWEET)
    tfi_wrds = util.do_wrd_tok(list(corpus_dct.values()))

if run_tfidf:
    wrd_freq = gsn.calc_tf(corpus2, word_tokens=True, calctyp="UNIQ")
    tws_wrd = gsn.count_tweets_for_word(wrd_freq)
    w_idf = gsn.calc_idf(wrd_freq, tws_wrd)
    tf_idf = gsn.calc_tf_idf(wrd_freq, w_idf)
    tfi_avg: dict[str, float] = gsn.calc_corpus_tfidf(tf_idf, calctyp="AVG")
    tfi_avg = {k: tfi_avg[k] for k in sorted(tfi_avg, key=lambda x: tfi_avg.get(x), reverse=True)}

if run_word_scoring:
    from statistics import mean, median, quantiles
    import tweet_visualize as gsviz

    tiles = quantiles(tfi_avg.values(), n=3)
    print(tiles)
    print(f"max tfidf value is: {max(tfi_avg.values())}")
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
    tw_scored = gsn.score_good_words(tw_sentplus, TFI_GREAT, TFI_GOOD, TFI_BAD)

    plt_cfg = gsviz.create_layout()
    fig_wrd = gsviz.plot_word_values(tw_scored, plyt=plt_cfg, appd="Workplace Dynamics")

    print(f" median wrdscore: {median([x['wrdscore'] for x in tw_sentplus])}")
    print(f" mean wrdscore: {mean([x['wrdscore'] for x in tw_sentplus])}")
    print(f" max wrdscore: {max([x['wrdscore'] for x in tw_sentplus])}")
    tiles = quantiles([x['wrdscore'] for x in tw_sentplus], n=3)
    print(tiles)

    topthirdwrd: list = []
    for tw in tw_sentplus:
        if tw['wrdscore'] >= 2.0:
            topthirdwrd.append(tw)

if save_dataset:
    # tw_cleansent, miss_user2 = gsutil.cleanup_tweet_records(tw_sentiment, users, chktyps=False)
    import datetime as dt
    for k, v in qt_merge.items():
        if 'date' in v and isinstance(v['date'], dt.datetime):
            v['date']: str = v['date'].strftime("%Y-%m-%d %H:%M")
    util.save_dict(recs=qt_merge, fname="quotetweets", desc="quote text by originID")

    getusers: list = sorted(miss_users, key=lambda x: miss_users.get(x), reverse=True)
    util.print_missing_for_postman(msng=getusers, outname="get_users8")

    util.save_dataset(recs=tw_sentplus, fname="WorkplaceDynamics")
    util.save_dict(recs=corpus_dct, fname='Workplace_wordtokens', desc="word token list")
    util.save_dict(recs=hash_list, fname="Work_hashtags", desc="Workplace tags")
