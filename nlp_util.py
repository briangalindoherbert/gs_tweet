# encoding=utf-8
"""
gs_nlp_UTIL is a complete set of utlity classes and methods to
facilitate processing of Tweets for nlp analysis.
includes extracting tweets from JSON files,
saving tweet data sets, text wrangling, tokenization,
and analysis like word cloud and tf/idf.
"""
import copy
import re
import datetime as dt
import json
from os import listdir
from os.path import isfile, join
from collections import OrderedDict

import pandas as pd
from nltk.tokenize import TweetTokenizer
from gs_data_dictionary import TWEET_RECORD, TW_DIR, OUTDIR, GS_HASH, GS_URL, GS_MENT, GS_CONTRACT, \
    GS_PAREN, JUNC_PUNC, GS_UCS2, GS_STOP, UTF_MEANING

def strip_nonascii_tweets(twlst):
    """
    simple decode-encode cycling of tweet text to check for western/english character set
    :param twlst: list of dict of tweets
    :return:
    """
    tmplst: list = []
    startlen: int = len(twlst)
    for tw in twlst:
        if tmp := str(tw['text']).encode('utf-8').decode('ascii', 'ignore'):
            if not re.search("nan", tw['text']):
                tw['text']: str = tmp
                tmplst.append(tw)
    print(f"  strip_nonascii started with {startlen}, ended with {len(tmplst)}")

    return tmplst

def get_file_list(jsondir: str=TW_DIR):
    """
    get_file_list returns all .json files in directory specified by 'jsondir'.
    :param jsondir: str with pathname to directory with tweet files
    :return: list of .json files
    """
    all_files = [f for f in listdir(jsondir) if isfile(join(jsondir, f))]
    prod_list: list = []
    fil_count: int = 0
    reject_count: int = 0
    for candidate in all_files:
        if candidate.endswith(".json"):
            fil_count += 1
            prod_list.append(candidate)
        else:
            reject_count += 1
    print("\n")
    outlst: list = [f"selecting json files in {jsondir}", f"Input: {str(fil_count)} .json files, "
                                                          f"{str(reject_count)} files rejected"]
    box_prn(outlst)

    return prod_list

def read_json_files(file_names, workdir: str=TW_DIR):
    """
    used for responses to both GET tweet and GET user APIs.
    iterates through input list of json files, reads each with retrieve_json, and creates
    and returns an aggregate list of results.
    :param workdir:
    :param file_names: list of filenames or string with one filename
    :return: list of tweets
    """
    if isinstance(file_names, str):
        tw_file: str = f"{workdir}{file_names}"
        print(f"  read_json_files retrieving from {file_names}")
        userrecs = retrieve_json(tw_file)
    elif isinstance(file_names, list):
        userrecs: list = []
        print(f"  read_json_files: reading records from files in {workdir}")
        cum_recs: int = 0
        for x in iter(file_names):
            tw_file: str = f"{workdir}{str(x)}"
            wip, cc = retrieve_json(tw_file)
            # aggregate tweets from multiple files into a single list
            userrecs.extend(iter(iter(wip)))
            cum_recs += cc

        print(f"    {cum_recs} records read from {len(file_names)} files")
    else:
        print(f"ERROR: read_json_files problem with parm {file_names}")
        return 13

    return userrecs

def load_tweet_archive(archive_f, workdir: str=OUTDIR, desc: str="tweets", twrec=None):
    """
    method reads the tweet dataset that is saved to file via save_dataset function
    :param archive_f: string with filename of dataset archive
    :param workdir: the output dir of this app
    :param desc: description of data loaded from file
    :param twrec: data dictionary for tweet record
    :return: list of tweets, each tweet is a dict of field-value pairs
    """
    if twrec is None:
        twrec = TWEET_RECORD
    archive_f: str = workdir + archive_f
    with open(archive_f) as fh:
        rawtext = json.load(fh)
    x: str = str(len(rawtext['results']))
    fh.close()
    print(f"\n  LOAD_TWEET_DATASET: {x} {desc} loaded from file \n")

    tmplst: list = []
    nan_fields: dict = {x: 0 for x in twrec}
    good_fields: dict = {x: 0 for x in twrec}
    bad_types: dict[type, int] = {None: 0, float: 0, str: 0}
    bad_fields: int = 0
    for tw in rawtext['results']:
        tmpdct: dict = {}
        for k, v in tw.items():
            if twrec[k] is dt.datetime:
                tmpdct[k]: dt.datetime = get_date(v, option='DT')
                good_fields[k] += 1
            elif type(v) is twrec[k]:
                tmpdct[k] = v
                good_fields[k] += 1
            else:
                bad_types[type(v)] += 1
                nan_fields[k] += 1
                bad_fields += 1

        tmplst.append(tmpdct)

    print(f"  total None/NaN/NaT data fields: {bad_fields}")
    for k, v in nan_fields.items():
        if v > 0:
            print(f" field {k} had {v} bad values")
    print("bad data types:")
    print(bad_types.items())
    print(f"\n    good values by field name:")
    print(good_fields.items())
    print(f"\n    -- processed {len(tmplst)} Tweets from file archive")

    return tmplst

def retrieve_json(tw_f: str, txtonly: bool = False):
    """
    method reads json file looking for 'results' or
    returns dict with 3 keys: results, next, and requestParameters
    :param tw_f: str name of file
    :param txtonly: boolean
    :return 'results' field plus record count
    """
    with open(tw_f, mode='rb') as fh:
        rawtext = json.load(fh)
        if 'results' in rawtext:
            x = len(rawtext['results'])
            dset = rawtext['results']
        elif 'data' in rawtext:
            x = len(rawtext['data'])
            dset = rawtext['data']
        fh.close()
    if not txtonly:
        return dset, x
    else:
        twttxt = [dset[y]['text'] for y in range(x)]
    return twttxt, x

def json_write(tweets: list, savefil):
    """
    called by save_recs to save batch of tweets. easy to serialize to json.
    using write mode w+ as this method is used to write one complete set to its own file
    :param tweets: list of dict, each tweet is set of key:vals
    :param savefil: str os file name
    :return: 0 if success
    """
    fh_j = open(savefil, mode='w+', encoding='utf-8', newline='')
    tmp_dct: dict = {"results": tweets}
    json.dump(tmp_dct, fh_j, separators=(',', ': '), skipkeys=True, ensure_ascii=False)

    return fh_j.close()

def save_dataset(recs: list, fname: str = "tweetarchive"):
    """
    save batch of tweets to json file, after scrub and filter.
    Output from this Fx can be used to assemble larger corpus with multiple topics

    :param fname:
    :param recs: list of dicts or list of lists for tweet corpus
    :return: 0 if success
    """
    size: str = str(len(recs))
    dnow: str = "_" + dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d")
    batchfile = f"{fname}_{size}{dnow}.json"
    tw_savef = join(OUTDIR, batchfile)
    tmplst: list = []
    for tw in recs:
        if 'rt_src_dt' in tw and type(tw['rt_src_dt']) is not str:
            tw['rt_src_dt']: str = dt.datetime.strftime(tw['rt_src_dt'], "%Y-%m-%d %H:%M")
        if 'qt_src_dt' in tw and type(tw['qt_src_dt']) is not str:
            tw['qt_src_dt']: str = dt.datetime.strftime(tw['qt_src_dt'], "%Y-%m-%d %H:%M")
        if 'sent' in tw and type(tw['sent']) is not str:
            tw['sent']: str = dt.datetime.strftime(tw['sent'], "%Y-%m-%d %H:%M")
        tmplst.append(tw)

    json_write(tmplst, tw_savef)
    print(f"gs_nlp_util.SAVE_DATASET: {size} tweets saved to {tw_savef} \n")

    return

def save_tojson(tweets, savefil: str, jpath: str=OUTDIR):
    """
    called by save_recs to save a large batch of tweets to file.
    easy to chunk any junk and serialize to json
    :param tweets: list of dict or dict of dict for tweets
    :param savefil: str with name of file
    :param jpath: path to output files, default=OUTDIR
    :return: 0 if success
    """
    fh_j = open(jpath + savefil, mode='a+', encoding='utf-8', newline='')
    json.dump(tweets, fh_j, separators=(',', ':'))
    print(f"\n save_tojson: saved {len(tweets)} records to json file")

    return fh_j.close()

def save_recs(recs: list, fnames: str = "tweetarchive"):
    """
    saves a batch of tweets to a json file, so that pre-processing steps don't have to
    be redone for a corpus. also facilitates aggregating many batches of tweets
    :param recs: list of dicts or list of lists for tweet corpus
    :param fnames: str for prefix of archive filename
    :return: 0 if success
    """
    if len(recs) >= 40:
        size: str = str(len(recs))
        batchdate: str = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d")
        tw_savef = f"{fnames}_{size}_{batchdate}.json"
        if isinstance(recs, dict):
            tmpsave: dict = {}
            for tid, tw in recs.items():
                # look for possible date fields and reformat as string for json compatability
                if 'date' in tw:
                    if type(tw['date']) is dt.datetime:
                        tw['date']: str = dt.datetime.strftime(tw['date'], "%Y-%m-%d %H:%M")
                    elif type(tw['date']) is not str:
                        tw.pop('date')
                # need to add same pop logic for below items for future flexibility!
                if 'rt_src_dt' in tw and type(tw['rt_src_dt']) is not str:
                    tw['rt_src_dt']: str = dt.datetime.strftime(tw['rt_src_dt'], "%Y-%m-%d %H:%M")
                if 'qt_src_dt' in tw and type(tw['qt_src_dt']) is not str:
                    tw['qt_src_dt']: str = dt.datetime.strftime(tw['qt_src_dt'], "%Y-%m-%d %H:%M")
                if 'sent' in tw and type(tw['sent']) is not str:
                    tw['sent']: str = dt.datetime.strftime(tw['sent'], "%Y-%m-%d %H:%M")
                tmpsave[tid] = tw
            save_tojson(tmpsave, tw_savef)
            print("gs_nlp_util.save_recs:  %s tweets saved to %s \n" % (size, tw_savef))
            return 0
        else:
            print("\n save_recs expected a dict of dict to be passed \n")
            return 1
    else:
        # if we don't have at least 100 tweets to save, something's wrong
        print("\n save_recs thinks input parm is too small (< 40) to archive! \n")
        return 1

def get_date(twcreated, option: str = 'S'):
    """
    returns a date object from tweet 'created_at' string passed to the function
    :param twcreated: str timestamp indicating when tweet was posted
    :param option: returns string date if 'S', else use 'DT' to return a datetime value
    :return dt.datetime object for option="DT" or str if option='S'
    """
    if str(twcreated[:4]).isnumeric():
        tmpdt = f"{twcreated[:4]}-{twcreated[5:7]}-{twcreated[8:10]} {twcreated[11:16]}"

        twdate: dt.datetime = dt.datetime.strptime(tmpdt, '%Y-%m-%d %H:%M')
    else:
        tmpdt = f"{twcreated[-4:]}-{twcreated[4:7]}-{twcreated[8:10]} {twcreated[11:16]}"

        twdate: dt.datetime = dt.datetime.strptime(tmpdt, '%Y-%b-%d %H:%M')

    if option == "DT":
        return twdate
    elif option == "S":
        return twdate.strftime('%Y-%m-%d')

def cleanup_tweet_records(twlst, usrs: dict, chktyps: bool= True):
    """
    clean up data using data type dictionary, also removes tweet if text does not contain
    ascii characters (an inexact way to strip non-english tweets from dataset
    See data dictionary - TWEET_RECORD for layout
    :param twlst: list of dict with named tweet fields
    :param usrs: dictionary of users to fill in missing names plus followers & friends
    :param chktyps: if True, checks data types against TWEET_RECORD data dict
    :return:
    """
    from pandas import isnull
    tmplst: list = []
    miss_users: dict = {}
    found_usr: int = 0
    rec_nouser: int = 0
    for tw in twlst:
        if tmp := str(tw['text']).encode('utf-8').decode('ascii', 'ignore'):
            # gets rid of non-ascii characters or entire tweets
            tw['text']: str = tmp
            tw['id']: str = str(tw['id'])
            if tw.get('userid') and not tw.get('uname'):
                if tw['userid'] in usrs:
                    tmpusr: dict = usrs[tw.get('userid')]
                    found_usr += 1
                    if 'uname' in tmpusr:
                        tw['uname'] = tmpusr['uname']
                    elif 'name' in tmpusr:
                        tw['uname'] = tmpusr['name']
                    if 'followers' in tmpusr:
                        tw['ufollow']: int = tmpusr['followers']
                        tw['ufriends']: int = tmpusr['friends']
                else:
                    rec_nouser += 1
                    if tw['userid'] not in miss_users:
                        miss_users[tw['userid']] = 1
                    else:
                        miss_users[tw['userid']] += 1
            # precautionary check - read fx was writing ID fields as lists in prior ver.
            if 'rply_id' in tw and isinstance(tw['rply_id'], list) and len(tw['rply_id']) == 1:
                tw['rply_id']: str = tw['rply_id'][0]
            if 'rt_id' in tw and isinstance(tw['rt_id'], list) and len(tw['rt_id']) == 1:
                tw['rt_id']: str = tw['rt_id'][0]
            if 'qt_id' in tw and isinstance(tw['qt_id'], list) and len(tw['qt_id']) == 1:
                tw['qt_id']: str = tw['qt_id'][0]
            # there are sometimes extracts that have only ID and Text fields...
            if 'sent' not in tw or isnull(tw['sent']):
                tw['sent']: str = '2022-06-01 08:00'
            elif not isinstance(tw['sent'], str):
                tw['sent']: str = dt.datetime.strftime(tw['sent'], "%Y-%m-%d %H:%M")

            if 'rt_src_dt' in tw and isnull(tw['rt_src_dt']):
                tw['rt_src_dt']: str = tw['sent']
            elif 'rt_src_dt' in tw and type(tw['rt_src_dt']) is not str:
                tw['rt_src_dt']: str = dt.datetime.strftime(tw['rt_src_dt'], "%Y-%m-%d %H:%M")
            if 'qt_src_dt' in tw and isnull(tw['qt_src_dt']):
                tw['qt_src_dt']: str = tw['sent']
            elif 'qt_src_dt' in tw and type(tw['qt_src_dt']) is not str:
                tw['qt_src_dt']: str = dt.datetime.strftime(tw['qt_src_dt'], "%Y-%m-%d %H:%M")
            if chktyps:
                for k, v in tw.items():
                    if TWEET_RECORD[k] is int and type(tw[k]) is not int:
                        # could add try-catch block to force str or float to int, fallback below
                        tw[k]: int = 0
                    elif TWEET_RECORD[k] is float:
                        tw[k]: float = round(float(v), ndigits=2)
            tmplst.append(tw)

    miss_users: dict = {k: miss_users[k] for k in sorted(miss_users, key=lambda x: miss_users.get(x), reverse=True)}
    print(f"\n  cleanup_tweet_records: start: {len(twlst)}  end: {len(tmplst)} ")
    print(f"             found user names:  {found_usr} ")
    print(f"           missing user names:  {len(miss_users)} for {rec_nouser} tweets")

    return tmplst, miss_users

def convert_datetime_to_str(twlst):
    """
    clean up data using data type dictionary See data dictionary - TWEET_RECORD for layout
    :param twlst: list of dict with named tweet fields
    :param twrec: data dictionary tweet record
    :return:
    """
    tmplst: list = []

    for tw in twlst:
        if not isinstance(tw['sent'], str):
            tw['sent']: str = dt.datetime.strftime(tw['sent'], "%Y-%m-%d %H:%M")
        if 'rt_src_dt' in tw and not isinstance(tw['rt_src_dt'], str):
            tw['rt_src_dt']: str = dt.datetime.strftime(tw['rt_src_dt'], "%Y-%m-%d %H:%M")
        if 'qt_src_dt' in tw and not isinstance(tw['qt_src_dt'], str):
            tw['qt_src_dt']: str = dt.datetime.strftime(tw['qt_src_dt'], "%Y-%m-%d %H:%M")
        tmplst.append(tw)

    print(f"\n  cleanup_after_read: began with {len(twlst)}  ended with {len(tmplst)} ")

    return tmplst

def parse_user_data(usrlst: list):
    """
    from the raw data read from file into a list, parse individual user attributes
    :param usrlst: list of dict
    :return: users
    """
    users: list = []
    usr_dct: dict = {}
    tw_total = len(usrlst)
    for tmp in usrlst:
        # for id, use either field id or id_str
        tmpdct: dict = {}
        if 'name' in tmp:
            tmpdct['name'] = tmp['name']
        if 'username' in tmp:
            tmpdct['uname'] = tmp['username']
        if 'public_metrics' in tmp:
            tmpdct['followers'] = tmp['public_metrics']['followers_count']
            tmpdct['friends'] = tmp['public_metrics']['following_count']
            tmpdct['tw_total'] = tmp['public_metrics']['tweet_count']

        if str(tmp['id']) in usr_dct:
            usr_dct[str(tmp['id'])] |= tmpdct
        else:
            usr_dct[str(tmp['id'])] = tmpdct
    print(f"\n  parse_user_data read in {len(usr_dct)} Twitter user records \n")

    return usr_dct

def get_fields_simple(tw_obj: list):
    """
    get_tweet_fields parses the mutli-level, dict and list structure of tweets to populate
    key:vals (text, date, hashes, u_mentions, and counts for quoted, retweeted,
    and replies). captures long text (>140 char) from extended Tweet if original is truncated.
    Twitters shows ellipsis (ucs \x2026 char) if tweet exceeds 140 chars.
    REFER TO TWEET_RECORD IN APP DATA DICTIONARY FOR FULL LAYOUT AND DTYPES
    :param tw_obj: [{},{}] i.e. a list of dicts
    :return: list of dict (dict for each tweet)
    """
    def do_mentions(tmp_lst, dct_wip: dict, rectyp: str):
        """
        inner function to parse user mentions, note inconsistency from Twitter endpoints
        :param tmp_lst: fragment of tweet input to parse
        :param dct_wip: dict record in process with get_fields_simple2
        :param rectyp: "arch" if parsing full archive record, "GET" for GET_Tweet endpoint
        :return: dct_wip: dict working copy of tw_dct
        """
        if 'mentions' in dct_wip and isinstance(dct_wip['mentions'], str):
            dct_wip['mentions'] = [dct_wip['mentions']]
        elif 'mentions' not in dct_wip:
            dct_wip['mentions']: list = []
        if isinstance(tmp_lst, list):
            x = len(tmp_lst)
            for y in range(x):
                if rectyp == "arch":
                    dct_wip['mentions'].append(tmp_lst[y]['screen_name'])
                elif rectyp == "GET":
                    dct_wip['mentions'].append(tmp_lst[y]['username'])
                else:
                    dct_wip['mentions'].append(tmp_lst)
        elif isinstance(tmp_lst, str):
            dct_wip['mentions'].append(tmp_lst)

        return dct_wip

    def do_urls(tmp_lst):
        """
        inner function to parse url info from Tweet sections
        :param tmp_lst: fragment of tweet input to parse
        :return:
        """
        x = len(tmp_lst)
        if isinstance(tmp_lst, list):
            if 'urls' in tw_dct:
                # convert field to list to append new values
                if isinstance(tw_dct['urls'], str):
                    tw_dct['urls'] = [tw_dct['urls']]
            else:
                tw_dct['urls'] = []
            for y in range(x):
                tw_dct['urls'].append(tmp_lst[y]['expanded_url'])
        return

    def do_hashes(tmp_lst, dct_wip: dict):
        """
        inner function to parse hashtag info from Tweet sections
        :param tmp_lst: fragment of tweet input to parse
        :param dct_wip: w-i-p tweet record from calling func
        :return: dct_wip: dict record for tweet in process
        """
        x = len(tmp_lst)
        if isinstance(tmp_lst, list):
            if 'hashes' in dct_wip and isinstance(dct_wip['hashes'], str):
                    dct_wip['hashes'] = [dct_wip['hashes']]
            else:
                dct_wip['hashes'] = []
            for y in range(x):
                if 'tag' in tmp_lst[y] and tmp_lst[y]['tag'] not in dct_wip['hashes']:
                    dct_wip['hashes'].append(str(tmp_lst[y]['tag']).lower())
                elif 'text' in tmp_lst[y] and tmp_lst[y]['text'] not in dct_wip['hashes']:
                    dct_wip['hashes'].append(str(tmp_lst[y]['text']).lower())
        return dct_wip

    tw_list: list = []
    tw_total = len(tw_obj)
    tcount: int = 0
    for twmp in tw_obj:
        # for id, use either field id or id_str
        tw_dct: dict = {'tdate': get_date(twmp['created_at'], option='S'),
                        'text': twmp['text'],
                        'ttime': twmp['created_at'][11:16]
                        }
        tw_dct['sent']: dt.datetime = get_date(twmp['created_at'], option='DT')
        if 'lang' in twmp:
            tw_dct['lang'] = twmp['lang']
        if 'id_str' in twmp:
            tw_dct['id'] = twmp['id_str']
        elif 'id' in twmp:
            tw_dct['id'] = twmp['id']
        if "user" in twmp:
            tw_dct['userid'] = twmp['user'].get('id_str')
            tw_dct['uname'] = twmp['user'].get('screen_name')
            tw_dct['ufollow'] = twmp['user'].get("followers_count")
            tw_dct['ufriends'] = twmp['user'].get("friends_count")
        elif twmp.get('author_id'):
            tw_dct['userid'] = twmp['author_id']
        if twmp.get('conversation_id'):
            tw_dct['conv'] = twmp['conversation_id']
        if 'quoted_status_permalink' in twmp:
            do_urls(twmp['quoted_status_permalink']['expanded'])
        # prefer screen name, but with user ID can lookup user name with call to users endpoint
        if twmp.get('in_reply_to_screen_name'):
            tw_dct = do_mentions(twmp['in_reply_to_screen_name'], tw_dct, rectyp="")
            tw_dct['rply_uname'] = twmp['in_reply_to_screen_name']
            tw_dct['rply_uid'] = twmp['in_reply_to_user_id_str']
        if twmp.get('in_reply_to_user_id_str'):
            tw_dct['rply_uid'] = twmp['in_reply_to_user_id_str']
        if twmp.get('in_reply_to_status_id_str'):
            if 'rply_id' in tw_dct:
                if not str(tw_dct['rply_id']).startswith(twmp['in_reply_to_status_id_str']):
                    print(f" ERROR with reply to tweet id {tw_dct['id']}")
            else:
                tw_dct['rply_id'] = twmp.get('in_reply_to_status_id_str')
        if twmp.get("retweet_count"):
            tw_dct['qrr'] = twmp['quote_count'] + twmp['reply_count'] +\
                            twmp['retweet_count']
            tw_dct['rply'] = twmp['reply_count']
            tw_dct['rtwt'] = twmp['retweet_count']
            tw_dct['quot'] = twmp['quote_count']
            tw_dct['fave'] = twmp['favorite_count']
        elif twmp.get("public_metrics"):
            tw_dct['qrr'] = twmp['public_metrics']['quote_count'] +\
                            twmp['public_metrics']['reply_count'] +\
                            twmp['public_metrics']['retweet_count']
            tw_dct['rply'] = twmp['public_metrics']['reply_count']
            tw_dct['rtwt'] = twmp['public_metrics']['retweet_count']
            tw_dct['quot'] = twmp['public_metrics']['quote_count']
            tw_dct['fave'] = twmp['public_metrics']['like_count']
        else:
            tw_dct['qrr'] = 0
            tw_dct['quot'] = 0
            tw_dct['rply'] = 0
            tw_dct['rtwt'] = 0
            tw_dct['fave'] = 0

        if 'extended_tweet' in twmp:
            tw_dct['text'] = twmp['extended_tweet']['full_text']
            if 'entities' in twmp['extended_tweet']:
                if 'hashtags' in twmp['extended_tweet']['entities']:
                    tmp_lst = twmp['extended_tweet']['entities']['hashtags']
                    tw_dct = do_hashes(tmp_lst, tw_dct)
                if 'urls' in twmp['extended_tweet']['entities']:
                    tmp_lst: list = twmp['extended_tweet']['entities']['urls']
                    do_urls(tmp_lst)
                if 'user_mentions' in twmp['extended_tweet']['entities']:
                    tmp_lst: list = twmp['extended_tweet']['entities']['user_mentions']
                    tw_dct = do_mentions(tmp_lst, dct_wip=tw_dct, rectyp="arch")
        if 'entities' in twmp:
            if 'hashtags' in twmp['entities']:
                tmp_lst = twmp['entities']['hashtags']
                tw_dct = do_hashes(tmp_lst, tw_dct)
            if 'urls' in twmp['entities']:
                tmp_lst: list = twmp['entities']['urls']
                do_urls(tmp_lst)
            if 'user_mentions' in twmp['entities']:
                tmp_lst = twmp['entities']['user_mentions']
                tw_dct = do_mentions(tmp_lst, tw_dct, "arch")
            elif 'mentions' in twmp['entities']:
                tmp_lst = twmp['entities']['mentions']
                tw_dct = do_mentions(tmp_lst, tw_dct, "GET")
        # see https://developer.twitter.com/en/docs/twitter-api/data-dictionary/example-payloads
        if 'includes' in twmp and 'users' in twmp['includes']:
            dctx: list = twmp['includes']['users']
            for tmp in zip(dctx, range(1)):
                tw_dct['userid'] = tmp['id']
                tw_dct['uname'] = tmp['username']
                tw_dct['ufollow'] = tmp['public_metrics']['followers_count']
                tw_dct['ufriends'] = tmp['public_metrics']['following_count']
        if "referenced_tweets" in twmp:
            for ref in twmp["referenced_tweets"]:
                refk = ref['type']
                refv: str = str(ref['id'])
                if refk == "quoted":
                    if 'qt_id' in tw_dct:
                        if not refv.startswith(tw_dct['qt_id']):
                            print(f" problem with qt_id for {refv}")
                    else:
                        tw_dct['qt_id']: str = refv
                elif refk == "replied_to":
                    if 'rply_id' in tw_dct:
                        if not str(tw_dct['rply_id']).startswith(refv):
                            print(f" ERROR with reply to tweet id {tw_dct['id']}")
                    else:
                        tw_dct['rply_id']: str = refv
                else:
                    if 'rt_id' in tw_dct:
                        if not refv.startswith(tw_dct['rt_id']):
                            print(f" problem with rt_id for {refv}")
                    else:
                        tw_dct['rt_id']: str = refv
        if 'retweeted_status' in twmp:
            if 'rt_id' in tw_dct:
                if not str(tw_dct['rt_id']).startswith(twmp['retweeted_status']['id_str']):
                    print(f" problem with rt_id in retweeted status: {tw_dct['rt_id']}")
            else:
                tw_dct['rt_id']: str = twmp['retweeted_status']['id_str']

            tw_dct['rt_src_dt']: dt.datetime = get_date(twmp['retweeted_status']['created_at'], option='DT')
            tw_dct['rt_quot']: int = twmp['retweeted_status']['quote_count']
            tw_dct['rt_rtwt']: int = twmp['retweeted_status']['retweet_count']
            tw_dct['rt_rply']: int = twmp['retweeted_status']['reply_count']
            tw_dct['rt_fave']: int = twmp['retweeted_status']['favorite_count']
            if 'extended_tweet' in twmp['retweeted_status']:
                tw_dct['text'] = twmp['retweeted_status']['extended_tweet']['full_text']
            if 'user' in twmp['retweeted_status']:
                twmp['rt_uid'] = twmp['retweeted_status']['user']['id_str']
                twmp['rt_uname'] = twmp['retweeted_status']['user']['screen_name']
                twmp['rt_ufollow']: int = twmp['retweeted_status']['user']['followers_count']
                twmp['rt_ufriends']: int = twmp['retweeted_status']['user']['friends_count']
        if 'quoted_status' in twmp:
            if 'qt_id' in tw_dct:
                if not str(tw_dct['qt_id']).startswith(twmp['quoted_status']['id_str']):
                    print(f" problem with qt_id in quoted status for {tw_dct['qt_id']}")
            else:
                tw_dct['qt_id']: str = twmp['quoted_status']['id_str']
            tw_dct['qt_text'] = twmp['quoted_status']['text']
            tw_dct['qt_src_dt']: dt.datetime = get_date(twmp['quoted_status']['created_at'], option='DT')
            tw_dct['qt_quot']: int = twmp['quoted_status']['quote_count']
            tw_dct['qt_rply']: int = twmp['quoted_status']['reply_count']
            tw_dct['qt_rtwt']: int = twmp['quoted_status']['retweet_count']
            tw_dct['qt_fave']: int = twmp['quoted_status']['favorite_count']
            if 'user' in twmp['quoted_status']:
                twmp['qt_uid'] = twmp['quoted_status']['user']['id_str']
                twmp['qt_uname'] = twmp['quoted_status']['user']['screen_name']
                twmp['qt_ufollow'] = twmp['quoted_status']['user']['followers_count']
                twmp['qt_ufriends'] = twmp['quoted_status']['user']['friends_count']
            if 'extended_tweet' in twmp['quoted_status']:
                tw_dct['qt_text'] = twmp['quoted_status']['extended_tweet']['full_text']
        elif "quoted_status_id_str" in twmp:
            tw_dct['qt_id']: str = twmp["quoted_status_id_str"]

        tw_list.append(tw_dct)
        tcount += 1

    print("%5d  Tweets parsed" % tcount, end="\n", flush=True)
    print("\n")
    return tw_list

def parse_raw_fields(tw_obj: list):
    """
    get_tweet_fields parses the mutli-level, dict and list structure of tweets to populate
    key:vals (text, date, hashes, u_mentions, and aggregate counts for quoted, retweeted,
    and replied). captures long text (>140 char) from extended record if truncated in original.
    Twitters shows ellipsis (ucs \x2026 char) if tweet exceeds 140 chars.
        TW_DCT key-value record:
        id         str,     a string of 15 numbers uniquely identifying tweet
        text       str,     text of tweet, draws from extended if >140 chars
        tdate      str,     'yyyy-mm-dd' when tweet was sent
        ttime      str,     'hh:mm' when tweet was sent
        sent       datetime64[ns],  parsed and formatted using datetime package
        conv_id    str,     conversation id links all replies plus original for a thread
        uname      str,     tweet author's '@ handle'
        userid    str,     user_id is passed as a long string of numbers
        ufollow    int,     number of users who follow the named user
        reply_uid  str,     user ID for a reply tweet
        reply_name            object
        domain                object
        entity                object
        hashes                object, list of hashtags (#<tag>) in tweet
        urls                  object, enhanced content for the tweet
        mentions              object, user names mentioned in tweet or replied-to
        qrr        int64,  sum of quote, retweet, reply counts
        fave       nt64,   count of times tweet was favorited, aka liked
        rply_id    str,    Tweet ID of original that current tweet is a reply to
        rt_id      str,    points to Original Tweet (in case of multiple RT's)
        rt_qrr               float64
        rt_fave              float64
        rt_srcname  str,  user who wrote original tweet
        rt_srcfollow int, number of followers of tweet originator
        rt_srcfriend int, number of friends of tweet originator
        qt_id                 object, reference ID for tweet that was quoted
        qt_text               object, text of quoted tweet response to an original tweet
        qt_orig      datetime[ns],  date-time original tweet was sent
        qt_qrr               float64, sum of quote, retweet, reply for *Original Tweet*
        qt_fave              float64, count of favorites/likes for *Original Tweet*
        qt_urls         not captured, available under root level 'quoted_status_permalink'

    :param tw_obj: [{},{}] i.e. a list of dicts
    :return: list of dict (dict for each tweet)
    """

    def do_mentions(tmp_lst, dct_wip: dict, rectyp: str = "arch"):
        """
        inner function to parse user mentions, note inconsistency from Twitter endpoints
        :param tmp_lst: fragment of tweet input to parse
        :param dct_wip: dict record in process with get_fields_simple2
        :param rectyp: "arch" if parsing full archive record, "GET" for GET_Tweet endpoint
        :return: dct_wip: dict working copy of tw_dct
        """
        x = len(tmp_lst)
        if isinstance(tmp_lst, list):
            if 'mentions' in dct_wip:
                # convert field to list to append new values
                if isinstance(dct_wip['mentions'], str):
                    dct_wip['mentions'] = [dct_wip['mentions']]
            else:
                dct_wip['mentions'] = []

            if isinstance(tmp_lst, str):
                dct_wip['mentions'].append(tmp_lst)
            else:
                for y in range(x):
                    if rectyp == "arch":
                        dct_wip['mentions'].append(tmp_lst[y]['screen_name'])
                    elif rectyp == "GET":
                        dct_wip['mentions'].append(tmp_lst[y]['username'])
        return dct_wip

    def do_urls(tmp_lst):
        """
        inner function to parse url info from Tweet sections
        :param tmp_lst: fragment of tweet input to parse
        :return:
        """
        x = len(tmp_lst)
        if isinstance(tmp_lst, list):
            if 'urls' in tw_dct:
                # convert field to list to append new values
                if isinstance(tw_dct['urls'], str):
                    tw_dct['urls'] = [tw_dct['urls']]
            else:
                tw_dct['urls'] = []
            for y in range(x):
                tw_dct['urls'].append(tmp_lst[y]['expanded_url'])
        return

    def do_hashes(tmp_lst, dct_wip: dict):
        """
        inner function to parse hashtag info from Tweet sections
        :param tmp_lst: fragment of tweet input to parse
        :param dct_wip: w-i-p tweet record from calling func
        :return: dct_wip: dict record for tweet in process
        """
        x = len(tmp_lst)
        if isinstance(tmp_lst, list):
            if 'hashes' in tw_dct:
                # convert str to list to append new values
                if isinstance(dct_wip['hashes'], str):
                    dct_wip['hashes'] = [dct_wip['hashes']]
            else:
                dct_wip['hashes'] = []
            for y in range(x):
                if 'tag' in tmp_lst[y]:
                    dct_wip['hashes'].append(tmp_lst[y]['tag'])
                else:
                    dct_wip['hashes'].append(tmp_lst[y]['text'])
        return dct_wip

    tw_list: list = []
    tw_total = len(tw_obj)
    tcount: int = 0

    for twmp in tw_obj:
        # for id, use either field id or id_str
        tw_dct: dict = {'tdate': get_date(twmp['created_at'], option='S'),
                        'text': twmp['text'],
                        'ttime': twmp['created_at'][11:16]
                        }
        tw_dct['sent']: dt.datetime = get_date(twmp['created_at'], option='DT')
        if 'lang' in twmp:
            tw_dct['lang'] = twmp['lang']
        if 'id_str' in twmp:
            tw_dct['id'] = twmp['id_str']
        elif 'id' in twmp:
            tw_dct['id'] = twmp['id']
        if "user" in twmp:
            tw_dct['userid'] = twmp['user'].get('id_str')
            tw_dct['uname'] = twmp['user'].get('screen_name')
            tw_dct['ufollow'] = twmp['user'].get("followers_count")
            tw_dct['ufriends'] = twmp['user'].get("friends_count")
        elif twmp.get('author_id'):
            tw_dct['userid'] = twmp['author_id']

        if twmp.get('conversation_id'):
            tw_dct['conv'] = twmp['conversation_id']
        # prefer screen name, but with user ID can lookup user name with call to users endpoint
        if twmp.get('in_reply_to_screen_name'):
            tw_dct = do_mentions(twmp['in_reply_to_screen_name'], tw_dct)
            tw_dct['rply_uname'] = twmp['in_reply_to_screen_name']
        elif twmp.get('in_reply_to_user_id_str'):
            tw_dct = do_mentions(twmp['in_reply_to_user_id_str'], tw_dct)
            tw_dct['rply_uid'] = twmp['in_reply_to_user_id_str']
        if twmp.get('in_reply_to_status_id_str'):
            if 'rply_id' in tw_dct:
                if not str(tw_dct['rply_id']).startswith(twmp['in_reply_to_status_id_str']):
                    print(f" ERROR with reply to tweet id {tw_dct['id']}")
            else:
                tw_dct['rply_id'] = twmp.get('in_reply_to_status_id_str')

        if twmp.get("retweet_count"):
            tw_dct['qrr'] = twmp['quote_count'] + twmp['reply_count'] +\
                            twmp['retweet_count']
            tw_dct['rply'] = twmp['reply_count']
            tw_dct['rtwt'] = twmp['retweet_count']
            tw_dct['quot'] = twmp['quote_count']
            tw_dct['fave'] = twmp['favorite_count']
        elif twmp.get("public_metrics"):
            tw_dct['qrr'] = twmp['public_metrics']['quote_count'] +\
                            twmp['public_metrics']['reply_count'] +\
                            twmp['public_metrics']['retweet_count']
            tw_dct['rpct'] = twmp['public_metrics']['reply_count']
            tw_dct['rtct'] = twmp['public_metrics']['retweet_count']
            tw_dct['qtct'] = twmp['public_metrics']['quote_count']
            tw_dct['fave'] = twmp['public_metrics']['like_count']
        else:
            tw_dct['qrr'] = 0
            tw_dct['fave'] = 0
        # if text >140 chars, up to 280 chars is stored under extended tweet
        if 'extended_tweet' in twmp:
            tw_dct['text'] = twmp['extended_tweet']['full_text']
            if 'entities' in twmp['extended_tweet']:
                if 'hashtags' in twmp['extended_tweet']['entities']:
                    tmp_lst = twmp['extended_tweet']['entities']['hashtags']
                    tw_dct = do_hashes(tmp_lst, tw_dct)
                if 'urls' in twmp['extended_tweet']['entities']:
                    tmp_lst: list = twmp['extended_tweet']['entities']['urls']
                    do_urls(tmp_lst)
                if 'user_mentions' in twmp['extended_tweet']['entities']:
                    tmp_lst: list = twmp['extended_tweet']['entities']['user_mentions']
                    tw_dct = do_mentions(tmp_lst, dct_wip=tw_dct, rectyp="arch")
        elif 'entities' in twmp:
            if 'hashtags' in twmp['entities']:
                tmp_lst = twmp['entities']['hashtags']
                tw_dct = do_hashes(tmp_lst, tw_dct)
            if 'urls' in twmp['entities']:
                tmp_lst: list = twmp['entities']['urls']
                do_urls(tmp_lst)
            if 'user_mentions' in twmp['entities']:
                tmp_lst = twmp['entities']['user_mentions']
                tw_dct = do_mentions(tmp_lst, tw_dct, "arch")
            elif 'mentions' in twmp['entities']:
                tmp_lst = twmp['entities']['mentions']
                tw_dct = do_mentions(tmp_lst, tw_dct, "GET")

        if "referenced_tweets" in twmp:
            for ref in twmp["referenced_tweets"]:
                refk = ref['type']
                refv: str = str(ref['id'])
                if refk == "quoted":
                    if 'qt_id' in tw_dct:
                        if not refv.startswith(tw_dct['qt_id']):
                            print(f" problem with qt_id for {refv}")
                    else:
                        tw_dct['qt_id'] = refv
                elif refk == "replied_to":
                    if 'rply_id' in tw_dct:
                        if not str(tw_dct['rply_id']).startswith(refv):
                            print(f" ERROR with reply to tweet id {tw_dct['id']}")
                    else:
                        tw_dct['rply_id'] = refv
                else:
                    if 'rt_id' in tw_dct:
                        if not refv.startswith(tw_dct['rt_id']):
                            print(f" problem with rt_id for {refv}")
                    else:
                        tw_dct['rt_id'] = refv

        if 'retweeted_status' in twmp:
            if 'rt_id' in tw_dct:
                if not str(tw_dct['rt_id']).startswith(twmp['retweeted_status']['id_str']):
                    print(f" problem with rt_id in retweeted status: {tw_dct['rt_id']}")
            else:
                tw_dct['rt_id']: str = twmp['retweeted_status']['id_str']
            tw_dct['rt_src_dt']: dt.datetime = get_date(twmp['retweeted_status']['created_at'], option='DT')
            tw_dct['rt_qrr']: int = twmp['retweeted_status']['quote_count'] +\
                               twmp['retweeted_status']['reply_count'] +\
                               twmp['retweeted_status']['retweet_count']
            tw_dct['rt_fave']: int = twmp['retweeted_status']['favorite_count']
            if 'extended_tweet' in twmp['retweeted_status']:
                tw_dct['text'] = twmp['retweeted_status']['extended_tweet']['full_text']
            if 'user' in twmp['retweeted_status']:
                twmp['rt_uname'] = twmp['retweeted_status']['user']['screen_name']
                twmp['rt_ufollow']: int = twmp['retweeted_status']['user']['followers_count']
                twmp['rt_ufriends']: int = twmp['retweeted_status']['user']['friends_count']
        if 'quoted_status' in twmp:
            if 'qt_id' in tw_dct:
                if not str(tw_dct['qt_id']).startswith(twmp['quoted_status']['id_str']):
                    print(f" problem with qt_id in quoted status for {tw_dct['qt_id']}")
            else:
                tw_dct['qt_id']: str = twmp['quoted_status']['id_str']
            tw_dct['qt_text'] = twmp['quoted_status']['text']
            tw_dct['qt_src_dt']: dt.datetime = get_date(twmp['quoted_status']['created_at'], option='DT')
            tw_dct['qt_qrr']: int = twmp['quoted_status']['quote_count'] +\
                               twmp['quoted_status']['reply_count'] +\
                               twmp['quoted_status']['retweet_count']
            tw_dct['qt_qtct']: int = twmp['quoted_status']['quote_count']
            tw_dct['qt_rtct']: int = twmp['quoted_status']['retweet_count']
            tw_dct['qt_rpct']: int = twmp['quoted_status']['reply_count']
            tw_dct['qt_fave']: int = twmp['quoted_status']['favorite_count']
            if 'user' in twmp['quoted_status']:
                twmp['qt_uname'] = twmp['quoted_status']['user']['screen_name']
                twmp['qt_ufollow'] = twmp['quoted_status']['user']['followers_count']
                twmp['qt_ufriends'] = twmp['quoted_status']['user']['friends_count']
            if 'extended_tweet' in twmp['quoted_status']:
                tw_dct['qt_text'] = twmp['quoted_status']['extended_tweet']['full_text']
        elif "quoted_status_id_str" in twmp:
            tw_dct['qt_id']: str = twmp["quoted_status_id_str"]

        tw_list.append(tw_dct)
        tcount += 1

    print("%5d  Tweets parsed" % tcount, end="\n", flush=True)
    print("\n")
    return tw_list

def get_batch_from_file(batchfil: str):
    """
    reads a batch of tweets that were saved to file with save_tojson
    :param batchfil: str with name of file to read
    :return: list of dict
    """
    with open(batchfil, mode='r', encoding='utf-8') as f_h:
        rawtext = json.load(f_h)
    return rawtext

def save_dict(recs: dict, fname: str = "hashtag_file", desc: str = "top hashtags"):
    """
    saves a dictionary to file, particularly for saving top hashtag-count dictionary.
    :param recs: dictionary to save to file
    :param fname: str for archive prefix
    :param desc: description of what is being saved
    :return: 0 if success
    """
    size_int: int = len(recs)
    size_str: str = str(size_int)
    tw_savef: str = OUTDIR + fname + "_" + size_str + ".json"
    with open(tw_savef, 'w') as f:
        f.write(json.dumps(recs))
        f.close()
        print(f"SAVE_DICT: {size_int} {desc} saved to {tw_savef} \n")
    return 0

def load_dict(fname: str = "hashtag_file_2585.json", desc: str = "hashtags"):
    """
    loads a dictionary from a json file, particularly for archived hashtags and word tokens
    :param fname: json file containing archived dictionary
    :param desc: description of records loaded from file
    :return: dictionary loaded from file
    """
    tw_savef: str = OUTDIR + fname
    with open(tw_savef) as f:
        hash_dict: dict = json.loads(f.read())
        f.close()
        size_str: str = str(len(hash_dict))
        print(f"LOAD_DICT: {size_str} {desc} loaded from {desc} file \n")
    return hash_dict

def scrub_text_field(tweetxt: str, del_entity: bool = False):
    """
    scrub_text performs numerous, sequential text removal and modification tasks,
    turn some cleaning off if resulting tweets lose context for nlp or sentiment.
    :param tweetxt: str from text field of tweet
    :param del_entity: boolean- if true strip all hashtag text not only '#'
    :return: str of tweet
    """
    if not isinstance(tweetxt, str):
        return
    # remove newlines in tweets, they cause a mess with many tasks
    tweetxt: str = tweetxt.replace("\n", " ")
    splitstr = tweetxt.split()
    cleanstr: str = ""
    for w in splitstr:
        # if not intentional all caps word, then make lower case
        if not str(w).isupper():
            w = str(w).lower()
        cleanstr = cleanstr + " " + w
    # convert select emojis to equivalent text, to facilitate sentiment analysis
    for k, v in UTF_MEANING.items():
        cleanstr = re.sub(k, v, cleanstr)
    # remove @user_mentions and URLS (still accessible from 'mentions')
    cleanstr = re.sub(GS_URL, "", cleanstr)
    if del_entity:
        # remove hashtags if strict=True
        cleanstr = re.sub(GS_HASH, "", cleanstr)
        cleanstr = re.sub(GS_MENT, "", cleanstr)
    else:
        # if strict is false, leave words from hashtag, still strips users
        cleanstr = re.sub(GS_HASH, "\g<1>", cleanstr)
        cleanstr = re.sub(GS_MENT, "", cleanstr)
    # remove periods, no need for sentence demarcation in tweet
    cleanstr = re.sub("\.", " ", cleanstr)
    # expand contractions using custom dict of contractions
    for k, v in GS_CONTRACT.items():
        cleanstr = re.sub(k, v, cleanstr)
    # often ucs-2 chars appear in english tweets, can simply convert some
    for k, v in GS_UCS2.items():
        cleanstr = re.sub(k, v, cleanstr)
    # remove punctuation- after meaning translations
    cleanstr = re.sub(JUNC_PUNC, "", cleanstr)
    # remove leading or trailing whitespace:
    cleanstr = cleanstr.strip()

    return cleanstr

def scrub_tweets(tw_batch: list, strict: bool = False):
    """
    simply iterates over list of tweets and calls scrub_text_field to do low level
    parsing of tweet text.
    :param tw_batch: list containing batch of tweets
    :param strict: flag, if true strips hashtags and user mentions and extended characters
    :return: dict with primary text now containing full text of tweet
    """
    fixed: list = []
    for atweet in tw_batch:
        if isinstance(atweet, dict):
            # EXPECTED course - send list of dict (each tweet record)
            if atweet.get('text'):
                atweet['text'] = scrub_text_field(atweet['text'], del_entity=strict)
            elif atweet.get('group_text'):
                # groupby_date Tweet content kept in group
                atweet['group_text'] = scrub_text_field(atweet['group_text'], del_entity=strict)
            if atweet.get('qt_text'):
                atweet['qt_text'] = scrub_text_field(atweet['qt_text'], del_entity=strict)
            fixed.append(atweet)
        elif isinstance(atweet, list):
            # each atweet is a list of sentence strings for tweet
            if isinstance(atweet[0], str):
                fixed.append([scrub_text_field(seg, del_entity=strict) for seg in atweet])
        else:   # assume isinstance(atweet, str):
            atweet = scrub_text_field(atweet, del_entity=strict)
            fixed.append(atweet)

    return fixed

def iter_scrub_todict(tw_batch: list, strict: bool = False):
    """
    simply iterates over list of tweets and calls scrub_text_field to do low level
    parsing of tweet text.
    :param tw_batch: list containing batch of tweets
    :param strict: flag, if true strips hashtags and user mentions and extended characters
    :return: dict of dict with primary key as ID of Tweet
    """
    scrubbed: dict = {}
    for twv in tw_batch:
        if isinstance(twv, dict):
            scrubbed[twv['id']] = twv.copy()
        else:
            print("ERROR: iter_scrub_todict requires a list of dict as first parameter!")
    out_d: dict = {}
    for twk, twv in scrubbed.items():
            # EXPECTED course - send list of dict (each tweet record)
            if twv.get('text'):
                twv['text'] = scrub_text_field(twv['text'], del_entity=strict)
            if twv.get('qt_text'):
                twv['qt_text'] = scrub_text_field(twv['qt_text'], del_entity=strict)
            out_d[twk] = twv

    return out_d

def remove_parens_stops(twlst: list, stop1: list = GS_STOP, stop2: list = None):
    """
    final pre-processing b/f sentiment or word tokenizing.  input is list of dict (pref)
    or list of list, or list of str.
    take text out of parens, convert to lower (if not all caps), and remove STOP words
    default 1st list=GS_STOP, can pass custom STOPs as stop1 or stop2

    :param twlst: list of tweets (each tweet a dict), but can be list of list or list of str
    :param stop1: default=GS_STOP from data dict, list of words to remove from tweets
    :param stop2: optional second list of stop words
    :return: list of scrubbed tweets
    """

    def do_paras(twstr: str) -> str:
        """
        inner function removes parentheses and moves text in parens to end of tweet
        :param twstr: str text of a single tweet
        :return: modified input str
        """
        parafound = re_paren.search(twstr)
        if parafound:
            paratext = parafound.group(1)
            twstr: str = re_paren.sub("", twstr).strip()
            twstr = f"{twstr} {paratext[1:-1]}"
        return twstr

    def do_lcase_stops(twstr: str) -> str:
        """
        inner Fx- unless input string is allcaps, lowercase it and run against stoplist
        :param twstr: str with text of one tweet
        :return: text of Tweet with case corrected and stopwords removed
        """
        tw_wrds = twstr.split()
        tmplst: list = []
        for w in tw_wrds:
            # retain ALL CAPS words, otherwise lower-case them
            if not str(w).isupper():
                w = str(w).lower()
            if w in stop1:
                continue
            if stop2 and w in stop2:
                continue
            tmplst.append(w)
        twstr = " ".join(list(tmplst))
        return twstr

    tw_clean: list = []
    re_paren = re.compile(GS_PAREN)
    for atweet in twlst:
        if isinstance(atweet, str):
            tw_text: str = do_paras(atweet)
            atweet = do_lcase_stops(tw_text)
        elif isinstance(atweet, dict):
            if atweet.get('text'):
                tw_text: str = do_paras(atweet['text'])
                atweet['text'] = do_lcase_stops(tw_text)
        elif isinstance(atweet, list):
            tw_text: str = " ".join([str(x) for x in atweet])
            atweet = do_paras(tw_text)
            atweet = do_lcase_stops(atweet)

        tw_clean.append(atweet)

    return tw_clean

def flatten_list_dict(twlist: list, addq: bool=True):
    """
    create flat list of tweet text from list of dict, add text for quoted tweet if option set.
    :param twlist: list of dict
    :param addq: boolean if True and original text found on quoted tweet, adds it in
    :return: templst list of str with tweet text
    """
    tmplst: list = []
    for twthis in twlist:
        if isinstance(twthis, dict):
            if clntx := str(twthis['text']).encode('utf-8').decode('ascii', 'ignore'):
                if twthis.get('qt_text'):
                    if clnqt := str(twthis['qt_text']).encode('utf-8').decode('ascii', 'ignore'):
                        if not clnqt.find("nan"):
                            tmplst.append(f"{clntx} {clnqt}")
                elif not pd.isna(clntx):
                    tmplst.append(clntx)
        elif isinstance(twthis, list):
            print("  ERROR: flatten_twdict expects to be passed a list of dict of tweets")

    return tmplst

def flatten_dict_dict(twdict: dict, addq: bool=True):
    """
    create dict with just tweet text from dict of dict, option to add quoted tweet text.
    :param twdict: dict of dict of tweets
    :param addq: boolean if True and original text found on quoted tweet, adds it in
    :return: templst list of str with tweet text
    """
    tmpd: dict = {}
    append_qt: int = 0
    for twk, twv in twdict.items():
        if ascitx := str(twv['text']).encode('utf-8').decode('ascii', 'ignore'):
            if addq and 'qt_text' in twv and len(twv['qt_text']) > 3:
                asciqt: str = twv['qt_text']
                append_qt += 1
                tmpd[twk] = f"{ascitx} {asciqt}"
            else:
                tmpd[twk] = ascitx

    print(f"\n    flatten_dict: appended quotes for {append_qt} tweets")
    print(f"    wrote flat file of {len(tmpd)} tweets")

    return tmpd

def keep_text_date_metrics_only(twlist: list, qtmrg: dict, addq: bool=True):
    """
    create flat list of tweet text from list of dict or list of list
    add original text for quoted tweet if option is set.
    :param twlist: list of dict
    :param qtmrg: list with all quote tweet text plus original text.
    :param addq: boolean if True and original text found on quoted tweet, adds it in
    :return: templst list of str with tweet text
    """
    print(f"\n    keep_text_metrics_only starting with {len(twlist)} tweets and {len(qtmrg)} quotes")
    tmplst: list = []
    for twthis in twlist:
        if isinstance(twthis, dict):
            # if no ascii chars after decode-encode, then record is skipped
            if clntx := str(twthis['text']).encode('utf-8').decode('ascii', 'ignore'):
                tmpdct: dict = {'text': clntx, 'id': twthis['id'], 'qrr': twthis['qrr'],
                'fave': twthis['fave'], 'tdate': twthis['tdate']}
                if twthis.get('qt_text'):
                    if clntx := str(twthis['qt_text']).encode('utf-8').decode('ascii', 'ignore'):
                        if not re.search("nan", clntx):
                            tmpdct['text']: str = tmpdct['text'] + " " + clntx
                            tmpdct['qrr'] += twthis['qt_qrr']
                            tmpdct['fave'] += twthis['qt_fave']
                if tmpdct:
                    tmplst.append(tmpdct)
        else:
            print("keep-text-and-metrics  needs to be passed a list of dicts (each dict a tweet)")

    tmpids: set = (tw['id'] for tw in tmplst)
    for twk, twv in qtmrg.items():
        if twk not in tmpids:
            twv['id'] = twk
            tmplst.append(twv)

    print(f"    {len(tmplst)} records written to lightweight Tweet dataset\n")

    return tmplst

def do_wrd_tok(tweet_lst: list):
    """
    do_wrd_tok tokenizes words from a tokenized sentence list, or a string
    if tweettokenizer is used, the following constants are available: strip_handles: bool,
    reduce_len: bool, preserve_case: bool,
    :param tweet_lst: list of list of strings to word tokenize
    :return: list of list of word tokens
    """
    import filter_and_enrich as gsta
    w_tok= TweetTokenizer(strip_handles=True, reduce_len=False, preserve_case=True)
    word_total: int = 0
    tw_total: int = 0
    wrd_tkn_lsts: list = []
    if isinstance(tweet_lst, list):
        for this_tw in tweet_lst:
            if isinstance(this_tw, list):
                wrd_tkn_lsts.append(w_tok.tokenize(" ".join([str(x) for x in this_tw])))
            elif isinstance(this_tw, str):
                wrd_tkn_lsts.append(w_tok.tokenize(this_tw))
            elif isinstance(this_tw, dict):
                wrd_tkn_lsts.append(w_tok.tokenize(this_tw['text']))
    if isinstance(tweet_lst, dict):
        wrd_tkn_lsts.extend(w_tok.tokenize(twv) for twv in tweet_lst)

    for tw in wrd_tkn_lsts:
        word_total += len(tw)
        tw_total += 1
    w_frequency: dict = gsta.get_word_freq(wrd_tkn_lsts)
    uniq_wrds: int = len(w_frequency)
    box_prn(f"word_tokenize: {word_total} words from {tw_total} Tweets, {uniq_wrds} unique words")

    return wrd_tkn_lsts

def do_stops(twlst: list, stop1: list=GS_STOP, stop2: list=None):
    """
    do_stops is preprocessing function to remove word tokens based on a stop list
    if sent a list of word-token lists, it returns lists of word tokens
    if sent a list of dict with a 'text' field in dict, returns cleaned text
    :param twlst: list of list, list of dict, or list of str for Tweets
    :param stop1: list of stop words, use GS_STOP or STOP_TWEET
    :param stop2: list of stop words
    :return: list of tweets with word tokens and stop words removed
    """
    clean_list: list = []
    wrds_removed: int = 0
    final_words: int = 0
    processed: int = len(twlst)
    is_tokenlist: bool = False
    for twis in twlst:
        if isinstance(twis, list):
            is_tokenlist = True
            twlen: int = len(twis)
            tmp_wrds: list = [cw for cw in twis if cw not in stop1]
            if stop2 is not None:
                tmp_wrds = [cw for cw in tmp_wrds if cw not in stop2]
            wrds_removed += (twlen - len(tmp_wrds))
            final_words += len(tmp_wrds)
            clean_list.append(tmp_wrds)
        else:
            if isinstance(twis, dict):
                twemp: list = str(twis['text']).split()
            else:                                # assume isinstance(twis, str)
                twemp: list = twis.split()

            twlen: int = len(twemp)
            tmp_wrds: list = [cw for cw in twemp if cw not in stop1]
            if stop2 is not None:
                tmp_wrds = [str(cw) for cw in tmp_wrds if cw not in stop2]
            wrds_removed += (twlen - len(tmp_wrds))
            final_words += len(tmp_wrds)
            clean_list.append(' '.join([str(cw) for cw in tmp_wrds]))

    print(f"   do_stops processed {processed} Tweets")
    if is_tokenlist:
        print("        reading from lists of word-tokens")
    else:
        print("   reading from text field in dict for each Tweet")
    print(f"        {wrds_removed} words were removed from dataset")
    print(f"        {final_words} words in final list")

    return clean_list

def get_dates_in_batch(tws):
    """
    for a list of tweets, capture dates included and count for each day
    :param tws: list of dict of tweets
    :return:
    """
    tw_days: dict = {}

    def do_dateparse(a_tw):
        """
        inner function to parse dates and times
        :param a_tw: a dict of the fields in a single tweet
        :return:
        """
        if isinstance(a_tw, dict) and a_tw.get('tdate'):
            tw_d = dt.datetime.strptime(a_tw['tdate'], '%Y-%m-%d')
            if tw_d in tw_days:
                if a_tw.get('ttime'):
                    tmp_hr = str(a_tw['ttime'][:2])
                    if tmp_hr in tw_days[tw_d]:
                        tw_days[tw_d][tmp_hr] += 1
                    else:
                        tw_days[tw_d][tmp_hr] = 1
                elif 'unk' in tw_days[tw_d]:
                    tw_days[tw_d]['unk'] += 1
                else:
                    tw_days[tw_d]['unk'] = 1
            else:
                tw_days[tw_d] = {}
                if a_tw.get('ttime'):
                    tw_days[tw_d][str(a_tw['ttime'][:2])] = 1
        return

    if isinstance(tws, list):
        for a_tw in tws:
            do_dateparse(a_tw)
    elif isinstance(tws, dict):
        for a_tw in tws.values():
            do_dateparse(a_tw)
    # sorting prior to printing
    tw_days: OrderedDict = OrderedDict([(k, tw_days[k]) for k in sorted(tw_days.keys())])
    return tw_days

def print_day_hour_distro(days_hrs: OrderedDict):
    """
    character-print function to show distribution of Tweet dataset by
    count for each day and hour.
    :param days_hrs: OrderedDict of dict key: date ->{2-digit hour: tweet count}
    :return: 0
    """
    print("Tweets per 1-hr block for each day: 2-digit hr - tweet count)")
    for seq_d, (d, dh) in enumerate(days_hrs.items(), start=1):
        print("Day %2d, Date: %10s" % (seq_d, dt.datetime.strftime(d, '%Y-%m-%d')))
        dh = {k: dh[k] for k in sorted(dh.keys())}
        days_hrs[d] = dh
        for hh, cnt in dh.items():
            print(f"  {str(hh)} hour: {str(cnt)} tweets | ", sep="", end="")
        print(" ")
    print("\n")
    return 0

def print_distro_bydate(days_hrs: OrderedDict):
    """
    character-print function to show distribution of Tweets by date
    :param days_hrs: OrderedDict of dict key: date ->{2-digit hour: tweet count}
    :return: 0
    """
    day_tot: int = 0
    tw_tot: int = 0
    print("Tweets per day in current dataset)")
    for seq_d, (d, dh) in enumerate(days_hrs.items(), start=1):
        xday: str = dt.datetime.strftime(d, '%Y-%m-%d')
        twcnt: int = sum(dh.values())
        tw_tot += twcnt
        day_tot = seq_d
        print(f" Day {seq_d} {xday}  tweet count: {twcnt}")
    print(f"\n    Dataset has {tw_tot} Tweets across {day_tot} total days")
    print("\n")
    return 0

def box_prn(message):
    """
    prints string or list of strings passed as 'message' parameter
    :param message: can be str or list of str
    :return: n/a
    """
    print("*" * 66)
    print("*" + " " * 64 + "*")
    if isinstance(message, str):
        bxpr_l(message)
    elif isinstance(message, list):
        for msgseg in message:
            bxpr_l(msgseg)
    print("*" + " " * 64 + "*")
    print("*" * 66)

    return

def bxpr_l(msg: str):
    """
    bxpr_1 is a small utility method to print message lines(s) for method box_prn
    :param msg:
    :return:
    """
    y = len(msg)
    while y > 61:
        brkpt: int = msg.find(" ", 53)
        if not 52 < brkpt < 62:
            brkpt = 62

        outseg: str = msg[:brkpt]
        fill: str = " " * (62 - len(outseg))
        print(f"* {outseg} {fill}*")
        msg = msg[brkpt:].strip()
        y = len(msg)

    fill: str = " " * (62 - len(msg))
    print(f"* {msg} {fill}*")

    return

def print_missing_for_postman(msng: list, outname: str= "missing", limit: int=800):
    """
    get the missing tweet ids for convenience to send to postman
    :param msng: list of missing tweets
    :param outname: str name for output file
    :param limit: maximum Tweet IDs to output
    :return None
    """
    import csv
    tmplst: list = []
    rply: int = 0
    qt: int = 0
    is_tweet: bool = False
    for x in msng:
        if 'missing_from' in x:
            is_tweet = True
            if str(x['missing_from']).startswith('rply'):
                tmplst.append(x['rply_id'])
                rply += 1
            elif str(x['missing_from']).startswith('qt'):
                tmplst.append(x['qt_id'])
                qt += 1
            elif str(x['missing_from']).startswith('rt'):
                tmplst.append(x['rt_id'])
        else:
            tmplst.append(x)

    if tmplst:
        outf: str = f"{OUTDIR}{outname}_{len(tmplst)}.csv"
        with open(outf, 'w+') as wh:
            csvptr = csv.writer(wh)
            csvptr.writerow([str(x) for x in tmplst])
            wh.close()
            if is_tweet:
                print(f"\n  print_missing_for_postman: {rply} reply ids and {qt} QT ids")
            print(f"  wrote {len(tmplst)} missing ID's to {outf}")



    return

def filter_listdict_by_dates(twl: list, startd: str=None, endd: str=None):
    """
    parses the tweet list of dict by searching for likely date fields and applying filter
    :param twl: list of dict
    :param startd: string of Y-m-d starting date
    :param endd: string of Y-m-d ending date
    :return:
    """
    tmplst: list = []
    strt_len: int = len(twl)
    for twx in twl:
        if isinstance(twx, dict) and twx.get('date'):
            dtmp = twx.get('date')
            if dtmp[:4] < startd[:4]:
                continue
            if dtmp[:4] == startd[:4]:
                if dtmp[5:7] < startd[5:7]:
                    continue
                if dtmp[5:7] == startd[5:7] and dtmp[8:10] < startd[8:10]:
                    continue
            if dtmp[:4] > endd[:4]:
                continue
            if dtmp[:4] == endd[:4]:
                if dtmp[5:7] > endd[5:7]:
                    continue
                if dtmp[5:7] == endd[5:7] and dtmp[8:10] > endd[8:10]:
                    continue
            tmplst.append(twx)
    print("filter listdict started with %d rows, ended with %d rows" % (strt_len, len(tmplst)))

    return tmplst

def filter_ds_start_end(twl: list, startd: str=None, endd: str=None):
    """
       parses the tweet list of dict by start and end dates
       :param twl: list of dict with 'sent' field expected in Y-m-d hh:mm format
       :param startd: string of Y-m-d starting date
       :param endd: string of Y-m-d ending date
       :return:
       """
    tmplst: list = []
    bad_ct: int = 0
    datestart = dt.datetime.strptime(startd, "%Y-%m-%d")
    dateend = dt.datetime.strptime(endd, "%Y-%m-%d")
    for tw in twl:
        if tw.get('sent'):
            if isinstance(tw['sent'], str):
                tw['sent']: dt.datetime = dt.datetime.strptime(tw['sent'], "%Y-%m-%d %H:%M")
            if isinstance(tw['sent'], dt.datetime):
                if datestart < tw['sent'] < dateend:
                    tmplst.append(tw)
                else:
                    bad_ct += 1

    print(f"\n    filter_listd_start_end cropped {bad_ct} records out of date range")
    print(f"    ended with {len(tmplst)} tweets")

    return tmplst

def cull_low_quality(twl: list, max_weakwrd: int=3, maxrepeat: int=4):
    """
    eliminates tweets with many junk words, many repeat words, or unparseable text
    :param twl: list of dict of tweet
    :param max_weakwrd: maximum num of weak or vulgar words in tweet
    :param maxrepeat: most times a single word can be repeated within a tweet
    :return list of filtered tweets
    """
    print(f"\n  cull_tweets starting with {len(twl)} tweets")
    junk_words: list = ['shitty', 'sucks', 'ass', 'fuck', 'programming', 'python',
                        'javascript', 'java', 'code', 'openings', 'now', 'hiring']
    tmplst: list = []
    twl_copy: list = copy.deepcopy(twl)
    twl_len: int = len(twl)
    wrd_repetition: int = 0
    weak_wrds: int = 0
    no_parse: int = 0
    for tw in twl_copy:
        # check for overly vulgar or ignorant tweets...
        if find_andcount(tw['text'], junk_words, minf=max_weakwrd):
            weak_wrds += 1
        else:
            # tokenize alphanumeric words then check for repetition of the same word
            tmptxt: list = list(set(re.split(r'\W+', tw['text'])))
            no_repeat: bool = True
            if rptlst := [x for x in tmptxt if str(x).isalpha()]:
                # only process tweets with 3 or more intelligible words
                if len(rptlst) > 2:
                    for x in rptlst:
                        if len(re.findall(x, tw['text'])) > maxrepeat:
                            no_repeat = False
                            wrd_repetition += 1
                            continue
                    if no_repeat:
                        tmplst.append(tw)
            else:
                no_parse += 1

    if twl_len - len(tmplst) > 0:
        print(junk_words)
        print(f"\n  cull_low_quality: {twl_len} candidate tweets in dataset")
        print("  will not munge input list")
        print(f"                    {weak_wrds} removed due to {max_weakwrd} or more junk words")
        print(f"                    {wrd_repetition} had a single word repeated {maxrepeat} or more times")
        print(f"                    {no_parse} failed parsing for distinct words")
        print("                    ----------")
        print(f"        ended with  {len(tmplst)} tweets")

    return tmplst

def filter_hashes(twd: dict, goodhsh: list, minct: int=1):
    """
    refactored utility to check for presence of 'good' tags given a dict for single tweet
    if 6 or more hashes found, filter also fails (junk tweets often filled with hashes!)
    :param twd: dict with keys=field names + values for a single tweet
    :param goodhsh: list of strings of good hashtags, lowercase and no '#' symbol
    :param minct: the minimum matches needed to return 'True' result
    :return: True if found, False if not found
    """
    hset: set = set(goodhsh)
    goodrec: bool = False
    countdwn: int = minct
    if 'hashes' in twd:
        if len(twd['hashes']) >= 6:
            return goodrec
        for x in twd['hashes']:
            if x in hset:
                countdwn += -1
                if not countdwn:
                    goodrec = True
                    break

        if countdwn and 'text' in twd:
            for x in goodhsh:
                if re.search(x, twd['text']):
                    countdwn += -1
                    if not countdwn:
                        goodrec = True
                        break
    return goodrec

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

def filter_tags_words(twlst, topicw: set, offtop: set, gdhsh: set, minct: int=2):
    """
    filter for tweets that contain at least one of a list of topicwords.  Run this function
    prior to applying Vader sentiment- to be sure sentiment scores are relevant to topic.
    :param twlst: list of dict of Tweets
    :param topicw: list of words for topic
    :param offtop: list of anti-words or off-topic words
    :param gdhsh: list of good on-topic hashtags
    :param minct: minimum number of topic words to match on
    :return: list of approved tweets, and list of eliminated tweets
    """
    # from tweet_data_dict import GOOD_HASHES, BAD_IDS
    print(f"\n    get_tweets_with_topicwords: {len(twlst)} candidate tweets")
    tmplst: list = []
    cutlst: list = []
    stepct: dict = {'offtopic': 0, 'topicwrdtwo': 0, 'goodhash': 0, 'unknown': 0}

    for tw in twlst:
        tw['text'] = tw['text'].lower()
        if not find_andcount(tw['text'], offtop, minf=2):
            thresh: int = 0
            if filter_hashes(tw, goodhsh=gdhsh):
                stepct['goodhash'] += 1
                tmplst.append(tw)
            elif find_andcount(tw['text'], findtxt=topicw, minf=minct):
                stepct['topicwrdtwo'] += 1
                tmplst.append(tw)
            else:
                stepct['unknown'] += 1
                cutlst.append(tw)
        else:
            stepct['offtopic'] += 1

    print(f"                         {stepct['goodhash']} had at least one topic hashtag")
    print(f"                         {stepct['topicwrdtwo']} had at least {minct} topic words")
    print(f"                                {stepct['offtopic']} filtered off-topic")
    print(f"                                {stepct['unknown']} unknown content- written to other list")
    print(f"                  final set has {len(tmplst)} tweets")

    return tmplst, cutlst

def filter_antitags(twlst, antiwrds: set, antihsh: set, minct: int=2, minhsh: int=1):
    """
    similar to filter_tags_words but looks for specific anti-topics for tweet removal
    anti-hashes defaults to 1, antiwords defaults to 2
    :param twlst: list of dict of Tweets
    :param antiwrds: list of words off topic
    :param antihsh: list of off-topic hashtags
    :param minct: minimum number of words to match on
    :param minhsh: minimum number of hashes to match on
    :return: list of approved tweets, and list of eliminated tweets
    """
    # from tweet_data_dict import GOOD_HASHES, BAD_IDS
    print(f"\n    filter anti-tags and words: {len(twlst)} candidate tweets")
    tmplst: list = []
    stepct: dict = {'offtopic': 0, 'badhash': 0}

    for tw in twlst:
        tw['text'] = tw['text'].lower()
        if filter_hashes(tw, goodhsh=antihsh, minct=minhsh):
            stepct['badhash'] += 1
        elif find_andcount(tw['text'], findtxt=antiwrds, minf=minct):
                stepct['offtopic'] += 1
        else:
                tmplst.append(tw)

    print(f"               {stepct['badhash']} had at least one anti-tag")
    print(f"               {stepct['offtopic']} filtered for at least {minct} off-topic words")
    print(f"               returned dataset has {len(tmplst)} tweets")

    return tmplst

def filter_by_dates(twl: list, startd: str=None, endd: str=None, dnam: str='sent'):
    """
    parses the tweet list of dict by searching for likely date fields and applying filter
    :param twl: list of dict
    :param startd: string of Y-m-d starting date
    :param endd: string of Y-m-d ending date
    :param dnam: use 'sent' column by default
    :return:
    """
    tmplst: list = []
    strt_len: int = len(twl)
    for twx in twl:
        if isinstance(twx, dict) and isinstance(twx.get(dnam), str):
            dtmp = twx.get(dnam)
            if dtmp[:4] < startd[:4]:
                continue
            if dtmp[:4] == startd[:4]:
                if dtmp[5:7] < startd[5:7]:
                    continue
                if dtmp[5:7] == startd[5:7] and dtmp[8:10] < startd[8:10]:
                    continue

            if dtmp[:4] > endd[:4]:
                continue
            if dtmp[:4] == endd[:4]:
                if dtmp[5:7] > endd[5:7]:
                    continue
                if dtmp[5:7] == endd[5:7] and dtmp[8:10] > endd[8:10]:
                    continue
            tmplst.append(twx)
    print("filter listdict started with %d rows, ended with %d rows" % (strt_len, len(tmplst)))

    return tmplst

def make_tags_lowercase(twl: list):
    """
    this makes it much easier to match on hashtag searches
    :param twl:
    :return:
    """
    retlst: list = []
    for x in twl:
        if 'hashes' in x:
            tmphash: list = []
            for y in x['hashes']:
                tmphash.append(str(y).lower())
            x['hashes']: list = tmphash
        retlst.append(x)

    return retlst
