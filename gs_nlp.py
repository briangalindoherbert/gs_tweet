# coding=utf-8
"""
methods for analyzing and visualizing Tweets which have been pre-processed.
includes tf-idf calculation:  there are variants to deriving both the tf and idf components
for a word, for determining overall tf*idf for word, and for applying it to a corpus such
as in determining threshold for unimportant words based on tf*idf

My take on variations with calculating TF:
raw frequency: simply the number of occurrences of the term
relative frequency: count of target word divided by total word count for document
distinct frequency: count of target word divided by total unique words in document
augmented frequency, count of target divided by count of top occurring word in doc.
"""
import re
from math import fsum, log
from statistics import mean, median, quantiles
from collections import OrderedDict
from string import punctuation
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamulticore import LdaMulticore
from gensim.test.utils import datapath
from gs_data_dictionary import NOT_ALPHA

import nlp_util as util
from gs_data_dictionary import OUTDIR

def calc_tf(sentlst, word_tokens: bool = False, calctyp: str = "UNIQ", debug: bool = False):
    """
    tf = occurrences of word to other words in each tweet.
    calctyp options: UNIQ- word occurrences/unique words, COUNT-occurrences/total word count,
    or TOP-word occurrences versus most frequent word
    :param sentlst: list of tweet str, dict or list
    :param word_tokens: is sentlist for tweets word-tokenized?
    :param calctyp: options: UNIQ words, COUNT-total words, or TOP- vs most frequent word
    :param debug: boolean if True prints status messages
    :return: dict of key=target word, val=frequency based on calctyp
    """

    def do_fq(sent):
        """
        calculates dict of key:word, value:word count from text of tweet
        :param sent: text of tweet
        :return: dict with word : word count
        """
        freq_table = {}
        wrds: list = sent.split() if isinstance(sent, str) else sent
        for w in wrds:
            w = re.sub(NOT_ALPHA, "", w).lower()
            if w in freq_table:
                freq_table[w] += 1
            else:
                freq_table[w] = 1
        return freq_table

    tf_table: dict = {}
    tw_tot = len(sentlst)
    if debug:
        print(f"calc_tf creating term frequency for {tw_tot} Tweets:")
    for x in range(tw_tot):
        if word_tokens:
            fq_dct = do_fq(sentlst[x])
            if debug and len(fq_dct) < 2:
                print(f"calc_tf {x} got {len(fq_dct)} distinct word tokens")
        else:
            if isinstance(sentlst[x], dict):
                tfdoc: str = sentlst[x]['text']
            elif isinstance(sentlst[x], list):
                tfdoc: str = " ".join([w for w in sentlst[x]])
            else:
                tfdoc: str = sentlst[x]
            fq_dct = do_fq(tfdoc)
            if debug and len(fq_dct) < 2:
                print(f"calc_tf {x} got {len(fq_dct)} distinct words")

        if fq_dct:
            if calctyp == "UNIQ":
                denom: int = len(fq_dct)
            elif calctyp == "TOP":
                denom: int = max(fq_dct.values())
            elif calctyp == "COUNT":
                denom: int = sum(fq_dct.values())
            else:
                print(f"calc_tf not a Valid parameter!: {calctyp}")
                break

            tf_tmp: dict = {}
            for word, count in fq_dct.items():
                tf_tmp[word] = count / denom
        else:
            tf_tmp: dict = {"": 0}
        tf_table[x] = tf_tmp

    return tf_table

def count_tweets_for_word(word_tf: dict):
    """
    identifies number of Tweets ('docs') in which target word occurs
    :param word_tf:
    :return: OrderedDict descending order of key=word : value=count Tweets where word appears
    """
    docs_per_word = {}
    for sent, tf in word_tf.items():
        for word in tf:
            if word in docs_per_word:
                docs_per_word[word] += 1
            else:
                docs_per_word[word] = 1

    w_descend: list = sorted(docs_per_word, key=lambda x: docs_per_word.get(x), reverse=True)
    w_docs: OrderedDict = {k: docs_per_word[k] for k in w_descend}

    return w_docs

def calc_idf(word_tf: dict, docs_word: OrderedDict):
    """
    idf = natural log of (total number of docs / docs in which target word occurs)
    :param word_tf: dict returned by calc_tf
    :param docs_word: OrderedDict returned by count_tweets_for_word
    :return: dict of dict, each tweet has dict of inverse document frequency values
    """
    idf_matrix = {}
    doc_count: int = len(word_tf)
    for sent, tf in word_tf.items():
        idf_table = {word: log(doc_count / int(docs_word[word])) for word in tf.keys()}

        idf_matrix[sent] = idf_table

    return idf_matrix

def calc_tf_idf(tf_matrix, idf_matrix):
    """
    creates dict of doc dict, each doc dict has key:val pais of: word : tf*idf value
    :param tf_matrix: returned by calc_tf
    :param idf_matrix: returned by calc_idf
    :return: dict of dict with word and tf*idf score for each doc in corpus
    """
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {word1: float(value1 * value2) for (word1, value1), (word2, value2) in
                        zip(f_table1.items(), f_table2.items())}

        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

def calc_corpus_tfidf(tfi_dct: dict, calctyp: str = "SUM"):
    """
    tf*idf is calculated at doc level, this calcs a single tf*idf per word at corpus level.
    my search found no standards for this. I decided to provide a SUM or AVERAGE option.
    sum adds value per occurrence, but is offset by idf penalty for commodity words.

    :param tfi_dct: the dict produced from calc_tf_idf
    :param calctyp: "SUM" or "AVG" to indicate how to aggregate values for target word
    :return: dict of words sorted by descending tfidf score
    """
    wrd_sc: dict = {}

    if calctyp == "AVG":
        for this in tfi_dct.values():
            for wrd, val in this.items():
                if wrd in wrd_sc:
                    wrd_sc[wrd].append(val)
                else:
                    wrd_sc[wrd] = [val]

        tmp_val: dict = {wrd: sum(vals) / len(vals) for wrd, vals in wrd_sc.items()}
        wrd_sc = tmp_val

    elif calctyp == "SUM":
        for this in tfi_dct.values():
            for wrd, val in this.items():
                wrd_sc[wrd] = wrd_sc[wrd] + val if wrd in wrd_sc else val
    srtd: list = sorted(wrd_sc, key=lambda x: wrd_sc.get(x), reverse=True)
    wrd_sc: OrderedDict = {k: wrd_sc[k] for k in srtd}

    return wrd_sc

def do_tfidf_stops(tf_dct: dict, cullthird: bool=True):
    """
    return words that are below median tfidf value, for a tfidf stop list
    :param tf_dct:
    :param cullthird: if true, culls the bottom third by tfidf score, else culls bottom half
    :return:
    """
    # tf_dct = {k: tf_dct[k] for k in tf_dct if str(k).isalpha()}
    tmpdct = {k: tf_dct[k] for k in sorted(tf_dct, key=lambda x: tf_dct[x], reverse=False)}
    rec_cnt: int = len(tmpdct)
    tfiavg = fsum(tmpdct.values()) / rec_cnt
    tfimedian = median(tmpdct.values())
    outdct: dict = {}
    print(f"  do_tfidf_stops:  calculating average and median values for {rec_cnt} Words")
    print(f"      average tfidf value: {tfiavg: .2f}")
    if cullthird:
        tiles = quantiles(tmpdct.values(), n=3, method="inclusive")
        for k, v in tmpdct.items():
            if v <= tiles[0]:
                outdct[k] = v
        print(f"        creating stop list from bottom third tf*idf value- {tiles[0]} ")
    else:
        med_record: int = int(round(rec_cnt/2, ndigits=0))
        outdct = {tw[0]: tw[1] for tw, cnt in zip(tmpdct.items(), range(med_record))}
        print(f"      creating stop list for words below {tfimedian:.2f} tfidf score")

    return outdct

def keep_tfidf_midhalf(tf_dct: dict):
    """
    return stoplist of words below 1st and above 3rd quartile.
    :param tf_dct:
    :return:
    """
    tmpdct = {k: tf_dct[k] for k in sorted(tf_dct, key=lambda x: tf_dct[x], reverse=False)}
    rec_cnt: int = len(tmpdct)
    tiles = quantiles(tmpdct.values(), n=4, method="inclusive")
    print(f"  keep_tfidf_midhalf: calculating for {rec_cnt} Words")
    outdct: dict = {k: v for k, v in tmpdct.items() if v < tiles[0] or v > tiles[2]}

    print(f"        creating stop list from bottom third tf*idf value- {tiles[0]} ")
    print(f"        stop list for words below {tiles[0]:.2f} or above {tiles[2]:.2f}")
    print(f"        final stopwords: {len(outdct)}")

    return outdct

def get_combined_toplist(qrrlst, favelst: list, sntlst: list):
    """
    combine qrr, fave and sentiment lists to identify most 'interesting' tweets.
    derived field: 'type' indicates if tweet from f=fave, q=qrr and/or s=sentiment
    :param sntlst: list of dict of high sentiment Tweets
    :param qrrlst: list of dict of high qrr count Tweets
    :param favelst: list of dict of high fave count Tweets
    :return: list of dict for combined top tweet list
    """
    new_list: list = []
    tops_lst: list = qrrlst
    sentids: list = sorted(k['id'] for k in sntlst)
    faveids: list = sorted(k['id'] for k in favelst)
    qrrids: list = sorted(k['id'] for k in qrrlst)
    print(f"\n get_combined_toplist starting from {len(sentids)} Tweets high qrr count")
    for x in tops_lst:
        if x['id'] not in faveids:
            x['type'] = 'qs' if x['id'] in sentids else 'q'
            new_list.append(x)
        else:
            x['type'] = 'qfs' if x['id'] in sentids else 'qf'
            new_list.append(x)

    for x in favelst:
        if x['id'] not in qrrids:
            x['type'] = 'fs' if x['id'] in sentids else 'f'
            new_list.append(x)

    tops_lst = sorted(new_list, key=lambda x: x.get('date'), reverse=False)
    qfs_lst: list = [x for x in tops_lst if len(x['type']) == 3]
    print("         --------")
    print(f"        {len(qfs_lst)} tweets meet all 3 score criteria \n")
    print("\n")

    return tops_lst, qfs_lst

def get_negsent_toplist(qrrlst, favelst: list, sntlst: list):
    """
    use results of get_pctle_qrr and get_pctle_fave to create 'most influential' tweets list
    adds derived key 'type' indicating if tweet had top f=fave, q=qrr or s=sentiment
    'type' allows selective plotting of tweets based on influence categorization.
    :param sntlst: list of dict of high sentiment Tweets
    :param qrrlst: list of dict of high qrr count Tweets
    :param favelst: list of dict of high fave count Tweets
    :return: list of dict for combined top tweet list
    """
    new_list: list = []
    tops_lst: list = sntlst
    sentids: list = sorted(k['id'] for k in sntlst)
    faveids: list = sorted(k['id'] for k in favelst)
    qrrids: list = sorted(k['id'] for k in qrrlst)
    print(f"\n get_negsent_toplist {len(sentids)} Tweets with high negative sentiment")

    for x in tops_lst:
        if x['id'] in faveids:
            x['type'] = 'qfs' if x['id'] in qrrids else 'fs'
            new_list.append(x)
        elif x['id'] in qrrids:
            x['type'] = 'qs'
            new_list.append(x)
    for x in qrrlst:
        if x['id'] not in sentids:
            x['type'] = 'qf' if x['id'] in faveids else 'q'
            new_list.append(x)
    tmpids: list = sorted(k['id'] for k in tops_lst)
    for x in favelst:
        if x['id'] not in tmpids:
            x['type'] = 'f'
            new_list.append(x)

    tops_lst = sorted(new_list, key=lambda x: x.get('date'), reverse=False)
    qfs_lst: list = [x for x in tops_lst if len(x['type']) == 3]
    print("         --------")
    print(f"        {len(qfs_lst)} tweets meet all 3 score criteria \n")
    print("\n")

    return tops_lst, qfs_lst

def final_toplist(twlst, topcut: float = 0.7):
    """
    selection of Tweet by qrr or fave count percentile
    :param twlst: list of dict with Tweets including qrr and fave counts and sent scores
    :param topcut: float (default=0.5) for decile qrr/fave rank of tweet
    :return: pd.DataFrame with highly selective toplist
    """
    twlen: int = len(twlst)
    quota: int = int(round((twlen + 1) * (1 - topcut), ndigits=0))
    qrrids: list = [x.get('id') for x in sorted(twlst, key=lambda x: int(x.get('qrr')), reverse=True)]
    qrrids = qrrids[:quota]
    favrecs: list = sorted(twlst, key=lambda x: int(x.get('fave')), reverse=True)
    favrecs = favrecs[:quota]
    toplst: list = list(sorted(twlst, key=lambda x: int(x.get('qrr')), reverse=True))
    toplst = toplst[:quota]
    for fid in favrecs:
        if fid['id'] not in qrrids:
            toplst.append(fid)

    toplst = sorted(toplst, key=lambda x: x.get('tdate'), reverse=False)
    qrrscore: list = [x.get('qrr') for x in toplst]
    favescore: list = [x.get('fave') for x in toplst]
    avgqrr: float = round(mean(qrrscore), ndigits=2)
    avgfave: float = round(mean(favescore), ndigits=2)

    if toplst:
        tmptxt: list = [f"final_toplist: {twlen} raw Tweets",
                        f"getting {topcut * 100:2.0f} percentile and above for Q-R-R or Likes",
                        f"QRR average= {avgqrr} , Like average= {avgfave} for selected"]
        util.box_prn(tmptxt)
        print(f"final toplist has {len(toplst)} tweets\n")

    return toplst

def get_hashtags(twlst):
    """
    return a hashtag dict by descending count of occurrence
    :param twlst: list of dict of tweets
    :return dict of hashtag : count
    """
    if isinstance(twlst[0], dict):
        hashlst: dict = {}
        hash_freq: int = 0
        total_tweets: int = len(twlst)
        for tw in twlst:
            if tw.get('hashes'):
                hash_found: bool = False
                for x in tw['hashes']:
                    x = str(x).lower()
                    if x in hashlst:
                        hashlst[x] += 1
                        hash_found = True
                    else:
                        if type(x) is str and len(x) > 2:
                            hashlst[x] = 1
                            hash_found = True
                if hash_found:
                    hash_freq += 1

        hashlst = {k: hashlst[k] for k in sorted(hashlst, key=lambda x: hashlst[x], reverse=True)}
        total_hash: int = len(hashlst)
        print(f"\n get_hashtags found hashtags in {hash_freq} of {total_tweets} tweets in dataset")
        print(f"                            and {total_hash} unique hashtags\n")
        return hashlst

def cleanup_for_cloud(twl: list):
    """
    makes sure input is word tokenized, words more than 3 chars and lowercase
    :param twl: list of word token lists
    :return:
    """
    tmplst: list = []
    for toklst in twl:
        if isinstance(toklst, str):
            toklst: list = str.split(toklst)
        cldtmp: list = []
        for wrd in toklst:
            if len(wrd) < 3:
                continue
            wrd = re.sub(NOT_ALPHA, "", wrd).lower()
            cldtmp.append(wrd)
        if cldtmp:
            tmplst.append(cldtmp)

    return tmplst

def prep_for_lda(twl: list):
    """
    preprocess cleaning for LDA topic modeling, uses string package punctuation
    :param twl: list of dict of tweets
    :return: list of str with scrubbed tweet text
    """
    stop = set(stopwords.words('english'))
    exclude = set(punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        """
        inner Fx to lowercase tweet, strip punctuation and remove stops
        :param doc: list of tweet text strings
        :return:
        """
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = "".join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean: list = []
    for tw in twl:
        if isinstance(tw, dict):
            doc_clean.append(clean(tw['text']).split())
        elif isinstance(tw, str):
            doc_clean.append(clean(tw).split())
        else:
            print("ERROR- prep_for_lda expects list of dict or list of str for Tweets")
            return None

    return doc_clean

def gensim_doc_terms(docs: list):
    """
    prepare gensim terms dictionary and document-terms matrix as intermediate step in
    topic-modeling of Tweet dataset.
    :param docs: list of tweets
    :return: list and gsm.Dictionary with term frequencies in Tweets and corpus
    """
    doc_dict: Dictionary = Dictionary(docs)
    doc_term_matrix: list = [doc_dict.doc2bow(doc) for doc in docs]

    return doc_term_matrix, doc_dict

def run_lda_model(doc_term, term_dict, topics: int = 4, chunk: int = 3000,
                  train_iter: int = 64, word_topics=True):
    """
    run training cycles with doc-term matrix and docterms dictionary
    parameters and meaning can be found at
        https://radimrehurek.com/gensim/models/ldamodel.html
    many parameters with interactions in call to instantiate Lda, explore to find best
    convergence and performance
    :param doc_term: doc-term matrix created in gensim_doc_terms
    :param term_dict: document term dictionary created in gensim_doc_terms
    :param topics: number of topics to compute and display
    :param chunk: number of documents used in each training 'chunk' def=2000
    :param train_iter: cycles of training to run in gensim
    :param word_topics: value for per_word_topics parm, True computes list of topics in descending order of prob
    :return:
    """
    Lda = LdaModel
    ldamodel: Lda = Lda(corpus=doc_term, num_topics=topics, id2word=term_dict,
                        chunksize=chunk, iterations=train_iter, passes=5,
                        update_every=0, per_word_topics=word_topics, alpha='auto',
                        eta=None, minimum_probability=0.01)

    return ldamodel

def display_lda(model: LdaModel, ntopic: int = 5):
    """
    shows results of topic modeling analysis with gensim
    :param model: instance of gsm.models.ldamodel.LdaModel
    :param ntopic: number of topics to display
    :return:
    """
    for x in range(ntopic):
        print(model.print_topic(topicno=x, topn=4))
        print("- - - - -")

    return None

def save_lda_model(ldam: LdaModel, mdl_f: str = "lda_model"):
    """
    save a gensim.models.ldamodel.LdaModel instance to a file
    :param ldam: gensim lda model
    :param mdl_f: file name to save as
    :return: None
    """
    temp_file = datapath(OUTDIR + mdl_f)
    ldam.save(temp_file)
    print("saved lda model to %s" % temp_file)

    return None

def load_lda(fq_mdl: str = "lda_model"):
    """
    load a pretrained gensim lda model from file
    :param fq_mdl: str filename of saved LDA model, default path is ./OUTDIR
    :return: lda model
    """
    Lda = LdaModel
    ldam = Lda.load(fq_mdl)

    return ldam

def test_text_with_model(new_txt: str, ldam: LdaModel, docterm):
    """
    get vectors for new content using our pretrained model
    :param new_txt: new tweet not part of training set
    :param ldam: trained LDA model
    :param docterm: gensim doc-term matrix
    :return: vectors for new content
    """
    vecs = ldam.update(new_txt, update_every=0)

    return vecs

def update_model_new_txt(ldam: LdaModel, wrd_tokns: list, ldadict: Dictionary):
    """
    run new content through LDA modeling
    :param ldam: gensim.models.LdaModel
    :param wrd_tokns: list of word tokens for new tweets
    :param ldadict: Dictionary of type gensim.corpora.Dictionary
    :return:
    """
    if isinstance(wrd_tokns, list):
        if isinstance(wrd_tokns[0], list):
            # each tweet is a list of strings within the larger list
            if isinstance(wrd_tokns[0][0], str):
                for twt in wrd_tokns:
                    new_corpus = [ldadict.doc2bow(wrd) for wrd in twt]
        elif isinstance(wrd_tokns[0], str):
            # each list element is text of tweet, need to word tokenize...
            for twt in wrd_tokns:
                tmp_tok: list = str(twt).split()
                new_corpus = [ldadict.doc2bow(wrd) for wrd in tmp_tok]

        new_content = new_corpus[0]
        vector = ldam[new_content]

    return vector

def get_model_diff(lda1: LdaModel, lda2: LdaModel):
    """
    get differences between pairs of topics from two models
    :param lda1:
    :param lda2:
    :return:
    """
    m1 = LdaMulticore.load(datapath(lda1))
    m2 = LdaMulticore.load(datapath(lda2))

    mdiff, annotation = m1.diff(m2)
    topic_diff = mdiff

    return topic_diff

def get_top_terms(lda: LdaModel, dDict: Dictionary, tpcs: int = 5, tpc_trms: int = 8):
    """
    map word index to actual word when printing top terms for topics
    :param lda: gensim.models.LdaModel instance
    :param dDict: gensim.corpora.Dictionary instance
    :param tpcs: number of topics for which to show terms
    :param tpc_trms: number of top terms to show for each topic
    :return:
    """
    tmpDict = dDict.id2token.copy()

    for x in range(tpcs):
        ttrm: list = lda.get_topic_terms(topicid=x, topn=tpc_trms)
        print("\ntop terms for topic %d:" % x)
        for twrd, val in ttrm:
            print("%s has probability %.3f" % (tmpDict[twrd], val))

    return None

def get_coherence(lda: LdaModel, dDict: Dictionary, wrdscln: list):
    """
    get coherence and perplexity scores for model
    :param lda: gensim models LdaModel
    :param dDict: gensim corpora Dictionary
    :param wrdscln: cleaned word tokens from corpus (filtered and lemmatized)
    :return:
    """
    coherence_lda = CoherenceModel(model=lda, texts=wrdscln, dictionary=dDict, coherence='c_v')
    # this call keeps blowing up!!!
    coherence_rslt = coherence_lda.get_coherence()
    print("Coherence score: %.4f" % coherence_rslt)

    return None

def check_tweet_POS(twx):
    """
    use spacy to check for parts of speech to filter for tweets with fully formed thoughts!
    DT Determiner
    EX Existential There. Example: “there is” … think of it like “there exists”)
    IN Preposition/Subordinating Conjunction.
    JJ Adjective.
    JJR Adjective, Comparative.
    JJS Adjective, Superlative.
    NN Noun, Singular.
    NNS Noun Plural.
    NNP Proper Noun, Singular.
    NNPS Proper Noun, Plural.
    PRP Personal Pronoun. Examples: I, he, she
    PRP$ Possessive Pronoun. Examples: my, his, hers
    RP Particle. Example: give up
    TO to. Example: go ‘to’ the store.
    VB Verb, Base Form. Example: take
    VBD Verb, Past Tense. Example: took
    VBG Verb, Gerund/Present Participle. Example: taking
    VBN Verb, Past Participle. Example: taken
    VBP Verb, Sing Present, non-3d take
    VBZ Verb, 3rd person sing. present takes
    :param twx:
    :return:
    """
    # import nltk
    # from nltk import word_tokenize
    from nltk import pos_tag

    tmplst: list = []
    for tw in twx:
        tagged = pos_tag(tw['text'].split())
        hasverb: bool = False
        hassubj: bool = False
        for x in tagged:
            if x[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'EX']:
                hasverb = True
            if x[1] in ['PRP', 'NN', 'NNS', 'NNP', 'NNPS']:
                hassubj = True

        if hasverb and hassubj:
            tmplst.append(tw)

    print(f"\n  check_tweet_POS started with {len(twx)} tweets")
    print("      verified presence of noun/pronoun and verb in each tweet")
    print(f"      ended with {len(tmplst)} after part-of-speech parsing")

    return tmplst

def getby_textortag(twl, twtxt: list, twtag: list, minfave: int=0):
    """
    select a dataset of tweets by words, phrases, or hashtags
    :param twl: expects a list of dict of tweets
    :param twtxt: list of words or phrases
    :param twtag: list of hashtags to search for (no '#' symbol)
    :return:
    """
    retlst: list = []
    nofave: int = 0
    foundwrd: int = 0
    foundtag: int = 0
    for tw in twl:
        if minfave and tw['fave'] < minfave:
            nofave += 1
            continue
        for wrd in twtxt:
            if re.search(wrd, tw['text']):
                foundwrd += 1
                retlst.append(tw)
                continue
        for tag in twtag:
            if 'hashes' in tw and tag in tw['hashes']:
                foundtag += 1
                if tw['id'] not in retlst:
                    retlst.append(tw)

    if foundwrd or foundtag or nofave:
        print(f"\n  getby_textortag searched {len(twl)} tweets")
        print(f"      {nofave} tweets had less than {minfave} likes")
        print(f"      {foundwrd} tweets matched on words or phrase")
        print(f"      {foundtag} tweets matched on hashtag")

    return retlst

def score_good_words(twl: list, GREAT: set, GOOD: set, BAD: set):
    """
    from top and mid-tier words from tfidf analysis, score each tweet's text,
    adding 2 for top words and 1 for mid-tier words.
    :param twl: list of dict of tweets
    :param GREAT: list of highest valued words on tfidf
    :param GOOD: list of mid-valued words on tfidf
    :return: list of tweets with new field for cumulative 'wrdscore'
    """
    returnlst: list = []
    for tw in twl:
        splits = tw['text'].split()
        neg_ct: int = 0
        tw['wrdscore']: int = 0
        for wrd in splits:
            if wrd in GREAT:
                tw['wrdscore'] += 2
            if wrd in GOOD:
                tw['wrdscore'] += 1
            if wrd in BAD:
                neg_ct += 1
        # minus one for every 3 weak words in tweet
        tw['wrdscore'] += -(neg_ct // 3)
        returnlst.append(tw)

    return returnlst
