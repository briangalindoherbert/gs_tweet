# encoding=utf-8
"""
a collection of functions related to processing Tweets and associated analytics within context
of a Pandas DataFrame.  Includes functions to apply scaling and scoring algorithms such
as logs, z-scores, and robust scaling.
"""
import datetime as dt
import pandas as pd
from math import log
from statistics import quantiles
import scipy.stats as stats
# from numpy.random import random

def prep_pandas(twdf: pd.DataFrame):
    """
    sorting and indexing prior to plotting with pandas
    :param twdf: pd.DataFrame of tweets
    :return: reformatted dataframes for both the above
    """
    dfcp: pd.DataFrame = twdf.copy(deep=True)
    snt_col: int = None
    for col in dfcp.columns:
        if col in ['sent', 'datetime']:
            snt_col = dfcp.columns.get_loc(col)
            if isinstance(dfcp.iat[0, snt_col], str):
                dfcp[col] = dfcp[col].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
                dfcp[col].astype('datetime64', copy=False, errors='ignore')
            dfcp.sort_values(by=[col], inplace=True, ignore_index=True)
            dfcp.reset_index(drop=True, inplace=True)
        elif col in ['date']:
            snt_col = dfcp.columns.get_loc(col)
            if isinstance(dfcp.iat[0, snt_col], str):
                dfcp[col] = dfcp[col].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
                dfcp[col].astype('datetime64', copy=False, errors='ignore')
            dfcp.sort_values(by=[col], inplace=True, ignore_index=True)
            dfcp.reset_index(drop=True, inplace=True)

    if snt_col:
        dtstr: str = dt.datetime.strftime(dfcp.iat[0, snt_col], "%Y-%m-%d %H:%M")
        print("first tweet on %s" %dtstr)
        twlen = len(dfcp)
        dtstr = dt.datetime.strftime(dfcp.iat[twlen - 1, snt_col], "%Y-%m-%d %H:%M")
        print("    last tweet on %s" % dtstr)

    return dfcp

def prep_scored_tweets(twdf: pd.DataFrame, logc: list, zc: list, strtyr:int=2021):
    """
    adjusts tweet score attributes to avoid problems with zero or negative values when
    applying scaling algorithms, then runs both standard and robust scaling on the data.
    :param twdf: pd.DataFrame with quoted/retweeted/reply and favorite counts plus sentiment
    :param logc: list of columns for log scaling
    :param zc: list of columns for z-scores and robust scaling
    :param strtyr: first year to include, in int format
    :return: twdf: PD.DataFrame with new scaled columns for counts/sentiment
    """

    twdf['year'] = twdf['tdate'].apply(lambda x: str(x)[:4])
    twdf['year'] = pd.to_numeric(twdf['year'], errors='coerce')
    twdf.drop(twdf.loc[twdf['year'] < strtyr].index, axis=0, inplace=True)
    if 'infl' not in twdf.columns:
        twdf['infl'] = twdf['qrr'] + twdf['fave']

    print(f"\n prep_scored_tweets calc of log and z-scores for {len(twdf)} tweets")
    sc1: int = 0
    twdf: pd.DataFrame = do_logscale(twdf, cnames=logc)
    print(f"    created log scale for {logc}")

    twdf: pd.DataFrame = rscale_col(twdf, zc)
    print(f"    created robust scaling for {zc}")
    print(twdf.describe())

    return twdf

def do_logscale(ldf: pd.DataFrame, cnames: list):
    """
    create log scoring for select dataframe columns
    :param ldf: pd.DataFrame
    :param cnames: list of columns for log scaling
    :return: pd.DataFrame
    """
    for column in cnames:
            ldf[column + '_log'] = ldf[column].apply(lambda x: round(log(x + 1), ndigits=1))

    return ldf

def rscale_col(zdf, colname):
    """
    applies robust scaling to one or more columns of a dataframe,
    :param zdf: pd.DataFrame
    :param colname: str or list of df column name(s)
    """
    if isinstance(colname, str):
        colname: list = [colname]

    for xcol in colname:
        denom: float = (zdf[xcol].quantile(0.75, interpolation="higher") - zdf[xcol].quantile(0.25,
                        interpolation="lower"))
        cmed = zdf[xcol].median()
        stdv = zdf[xcol].std()
        zdf[xcol + "_rsc"] = round(zdf[xcol].apply(lambda x: (x - cmed)/denom), ndigits=2)
        zdf[xcol + "_zsc"] = round(stats.zscore(zdf[xcol]), ndigits=2)
        # zdf[xcol + "_zscl"] = round((zdf[xcol] - zdf[xcol].mean()) / stdv, ndigits=2)

    return zdf

def aggreg_tweets_bydate(trades: pd.DataFrame, tdf: pd.DataFrame):
    """
    creates warehouse of summary by day of number of tweets in dataset, influence metrics,
    and average pos, neg, and compound sentiment

    :param trades: pd.Dataframe of market info- to give us date ranges for summary
    :param tdf: pd.Dataframe of tweets, which provides us data for roll-ups
    :return: pd.Dataframe with roll-up data
    """
    xrng: int = (trades.date.max() - trades.date.min()).days
    strtd: dt.datetime = trades.date.min()
    offset: dt.timedelta = dt.timedelta(days=1)

    curdt: dt.datetime = strtd
    dw_lst: list = []
    for x in range(xrng):
        dt_str: str = dt.datetime.strftime(curdt, "%Y-%m-%d")
        recs: int = int(tdf.loc[tdf['date'] == dt_str, ['id']].count(axis=0))
        if not recs >= 1:
            recs = 0
        negS: float = float(tdf.loc[tdf['date'] == dt_str, ['neg']].sum())
        posS: float = float(tdf.loc[tdf['date'] == dt_str, ['pos']].sum())
        compS: float = float(tdf.loc[tdf['date'] == dt_str, ['compound']].sum())
        qrrS: int = int(tdf.loc[tdf['date'] == dt_str, ['qrr']].sum())
        faveS: int = int(tdf.loc[tdf['date'] == dt_str, ['fave']].sum())
        tmpdct: dict = {'date': dt_str, 'count': recs, 'neg': negS,
                        'pos': posS, 'comp': compS, 'qrr': qrrS, 'fave': faveS}
        dw_lst.append(tmpdct)
        curdt += offset

    return dw_lst

def create_dataframe(twlist: list, dcol: str = "sent"):
    """
    create a pandas dataframe from a list of dicts, where each dict is one tweet
    :param twlist: list of dict for tweets after pre-processing
    :param dcol: optional name of date column to use in this table
    :return: pd.DataFrame from input list
    """
    df: pd.DataFrame = pd.DataFrame.from_records(twlist)
    print(f"\n create_dataframe: loading {len(twlist)} Twitter rows to Pandas")
    if dcol in df.columns:
        print(f"    {dcol} being used as date column in table")
    else:
        print(f"ERROR: {dcol} not found in table")
        return None
    dcol_num: int = df.columns.get_loc(dcol)
    if isinstance(df.iat[0, dcol_num], str):
        if len(df.iat[0, dcol_num]) == 16:
            df[dcol] = df[dcol].apply(lambda x: pd.to_datetime(x))
            df[dcol] = df[dcol].astype('datetime64[ns]')
        elif len(df.iat[0, dcol_num]) == 10:
            df[dcol] = df[dcol].apply(lambda x: pd.to_datetime(x))
            df[dcol] = df[dcol].astype('datetime64[ns]')
        else:
            print(f"ERROR: could not convert {dcol} column to date")
            return None
        dt.datetime.strftime(df[dcol].max(), "%Y-%m-%d")

    df.sort_values(dcol, inplace=True, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    strt: str = pd.to_datetime(df.sent.min())
    endd: str = pd.to_datetime(df.sent.max())
    print(f"    date spans from {strt} to {endd}\n")

    return df

def crop_df_to_date(twdf: pd.DataFrame, strtd: str = None, endd: str = None):
    """
    pass a start date and/or end date to crop a dataframe,
    handy to use prior to plotting a slice of a dataset.
    this Fx only processes valid dates provided- as "YYYY-mm-dd HH:MM" format string
    :param twdf: pd.DataFrame
    :param strtd: str as "YYYY-mm-dd HH:MM"
    :param endd: str as "YYYY-mm-dd HH:MM"
    :return: modified pd.DataFrame
    """
    tmpdf: pd.DataFrame = twdf.copy(deep=True)
    pre_len: int = len(tmpdf)
    print("\nCROP_DF_TO_DATE: trimming dataset to start and end dates")
    print("    starting with %d tweets" % pre_len)

    if 'sent' in tmpdf.columns:
        foundcol: int = tmpdf.columns.get_loc('sent')
        if isinstance(tmpdf.iat[0, foundcol], dt.datetime):
            # can also use below, but technically not a legit Fx call
            # if is_dt64(tmpdf['sent']):
            usethis: str = 'sent'
        elif isinstance(tmpdf.iat[0, foundcol], str):
            tmpdf['sent'] = tmpdf['sent'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
            tmpdf['sent'].astype('datetime64', copy=False, errors='ignore')
            usethis: str = 'sent'
        print("    using SENT column to remove dates outside of range...")

    elif 'date' in tmpdf.columns:
        foundcol: int = tmpdf.columns.get_loc('date')
        if isinstance(tmpdf.iat[0, foundcol], dt.datetime):
            usethis: str = 'date'
        else:
            if isinstance(tmpdf.iat[0, foundcol], str):
                tmpdf['date'] = tmpdf['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
                tmpdf['date'].astype('datetime64', copy=False, errors='ignore')
                usethis: str = 'date'
        print("    using DATE column to remove dates outside of range...")

    elif 'datetime' in tmpdf.columns:
        foundcol: int = tmpdf.columns.get_loc('datetime')
        if isinstance(tmpdf.iat[0, foundcol], dt.datetime):
            usethis: str = 'datetime'
        else:
            if isinstance(tmpdf.iat[0, foundcol], str):
                tmpdf['datetime'] = tmpdf['datetime'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
                tmpdf['datetime'].astype('datetime64', copy=False, errors='ignore')
                usethis: str = 'datetime'
        print("    using DATETIME column to remove dates outside of range...")

    if strtd:
        try:
            if len(strtd) == 10:
                # dt_strt: dt.date = dt.date.fromisoformat(strtd)
                dt_strt: dt.datetime = dt.datetime.strptime(strtd, "%Y-%m-%d")
            elif len(strtd) == 16:
                dt_strt: dt.datetime = dt.datetime.strptime(strtd, "%Y-%m-%d %H:%M")
            else:
                print("crop_df_to_date ERROR need 10 char date or 16 char date-time")
                return None
            tmpdf = tmpdf.loc[tmpdf[usethis] >= dt_strt,]
            print("    removing rows with dates prior to %s" % strtd)
        except ValueError:
            print("crop_df_to_date ERROR: invalid start date parameter")
            return None

        tmpdf.reset_index(drop=True, inplace=True)
    if endd:
        try:
            if len(endd) == 10:
                dt_end: dt.date = dt.datetime.strptime(endd, "%Y-%m-%d")
            elif len(endd) == 16:
                dt_end: dt.datetime = dt.datetime.strptime(endd, "%Y-%m-%d %H:%M")
            else:
                print("crop_df_to_date ERROR: need 10 char date or 16 char date-time")
                return None
            tmpdf = tmpdf.loc[tmpdf[usethis] <= dt_end,]
            print("    removing rows with dates later than %s" % endd)
        except ValueError:
            print("crop_df_to_date ERROR: invalid end date parameter")

    print("        %d records remain after removing %d rows\n"
          % (len(tmpdf), len(tmpdf) - pre_len))
    tmpdf.sort_values(usethis)
    tmpdf.reset_index(drop=True, inplace=True)

    return tmpdf

def fix_pandas_dtypes(df: pd.DataFrame):
    """
     tweetdf.dtypes
date                  object
text                  object
sent_time             object
sent          datetime64[ns]
id                    object
conv_id               object
uname                 object
user_id               object
reply_uid             object
reply_name            object
rply_id               object
ref_tweets            object
reply_name            object
domain                object
entity                object
hashes                object
urls                  object
mentions              object
qrr                    int64
fave                   int64
rt_id                 object
rt_qrr               float64
rt_fave              float64
rt_srcname
rt_srcfollow
rt_srcfriends
qt_id                 object
qt_text               object
qt_qrr               float64
qt_fave              float64
qt_srcname
qt_srcfollow
qt_srcfriends

    :param df: pandas DataFrame with Tweets plus analytics fields
    :return: pd.DataFrame
    """
    df.qt_src_dt.fillna()
    df.name.fillna(' ', inplace=True)
    df.name.fillna(' ', inplace=True)
    df.reply_uid.fillna(' ', inplace=True)
    df.reply_to.fillna(' ', inplace=True)

    # tweetdf.rt_id.fillna(" ")
    df = df.astype(dtype={'hashes': object, 'urls': object})

    df['rt_id'] = df['rt_id'].astype('str')
    df['rt_id'] = df['rt_id'].astype('str')

    df.rt_qrr.convert_dtypes(infer_objects=True)
    df.rt_fave.convert_dtypes(infer_objects=True)
    df.qt_qrr.convert_dtypes(infer_objects=True)
    df.qt_fave.convert_dtypes(infer_objects=True)

    df['rt_qrr'] = df['rt_qrr'].fillna(0)
    df['rt_fave'] = df['rt_fave'].fillna(0)
    df['qt_qrr'] = df['qt_qrr'].fillna(0)
    df['qt_fave'] = df['qt_fave'].fillna(0)

    filt_lst: list = df.to_dict("records")

    return df

def set_dist_sent_coding(df: pd.DataFrame):
    """
    sets a new column 'dist_clr' to set a color for originals, replies, RTs, and QTs
    :param df: a dataframe being prepped for 3d plotting, where color gradients are needed
    :return:
    """
    quant4: list = quantiles(df.neu)
    def quart_codes(x):
        if x < quant4[0]:
            return 1
        elif x < quant4[1]:
            return 2
        elif x < quant4[2]:
            return 3
        else:
            return 4

    df['distcode']: int = 1
    df['distcode']: int = df.rply_id.apply(lambda x: 2 if x and len(x) > 3 else x)
    df['distcode']: int = df.rt_id.apply(lambda x: 3 if x and len(x) > 3 else x)
    df['distcode']: int = df.qt_id.apply(lambda x: 4 if x and len(x) > 3 else x)

    df['sentcode']: int = df.neu.apply(lambda x: quart_codes(x))

    return df
