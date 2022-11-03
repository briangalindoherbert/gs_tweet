"""
this is an approach to training a gensim model that I used with musical lyrics, and am
now adapting for use with tweets.

"""

import gensim
import re
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer
from gs_data_dictionary import QUOTEDASH_TABLE, CHAR_CONVERSION, GS_CONTRACT, STOP_PREP, TW_DIR

unic_conv_table = str.maketrans(CHAR_CONVERSION)
punc_tbl = str.maketrans({key: None for key in "[];:,.?!*$&@%<>'(){}"})
re_brkttxt = re.compile(r"\[.+\]")
lemma = WordNetLemmatizer()

def do_skip_gram(iter_tw, batchsiz: int = 5000, passes: int = 5, thrds: int = 4, maxv: int = 600,
                 dim: int = 64, grpsz: int = 5, skipg: int = 1):
    """
    train word vectors based on skip gram, some parms that are settable with word2vec
    but are not settable from this Fx's parms: hs- if 1 use hierarchical softmax,
    default is negative sampling- see w2v 'ns' parm,
    max_vocab_size - while training- as opposed to max_final_vocab for output model
    :param iter_tw: batches of tweets provided by an iterator
    :param batchsiz: words provided per cycle to worker instances, typically 5-10k
    :param passes: number of passes (epochs) through the corpus, default=3
    :param thrds: number of concurrent processes to run (multi-processing threads)
    :param maxv: maximum final vocabulary for model
    :param dim: dimensionality of resulting word vectors (vector_size)
    :param grpsz: max distance from current to predicted word, typically 1-7
    :param skipg: 1 selects skipgram, 0 or other selects cbow
    :return:
    """
    #  sample=1e-3
    skipg_mdl = Word2Vec(sentences=iter_tw, batch_words=batchsiz, epochs=passes,
                         vector_size=dim, window=grpsz, workers=thrds, sg=skipg, hs=1,
                         max_final_vocab=maxv)

    return skipg_mdl

def text_cleanup(txt: str, word_ct: int, wrd_tok: bool=True):
    """
    performs common formatting for lines of text including the following:
    1. convert to lower case, remove non-ascii, remove punctuation
    2. expand contractions as per GS_CONTRACT from data_dict file
    3. convert common extended ascii/unicode chars like accented quote marks or ellipsis
    4. word tokenize the text and return it
    :param txt: string to be wrangled
    :param word_ct: int running total of words, will be incremented and returned by Fx
    :param wrd_tok: bool default to return list of word tokens, else return a string
    :return: tokenized list of words
    """
    if isinstance(txt, str):
        tmp = txt.encode('utf-8').decode('ascii', 'ignore').lower()
        tmp = txt.translate(unic_conv_table).translate(QUOTEDASH_TABLE)
        tmp = re.sub(r"\n", repl=r"", string=tmp)
        for wrd, expand in GS_CONTRACT.items():
            tmp = re.sub(wrd, repl=expand, string=tmp)
        tmptrans = tmp.translate(punc_tbl)
        splts: list = tmptrans.split()
        linetok: list = [w for w in splts if w not in STOP_PREP]
        word_ct += len(linetok)
    else:
        print("line cleanup expects an input string")
        linetok = []
        word_ct = 0

    if wrd_tok:
        return linetok, word_ct
    else:
        tmp = " ".join([w for w in linetok])
        return tmp, word_ct

def feed_tweets(prefix, twl: list):
    """
    this generator returns each non-blank text of a tweet as stream to the object
    it calls line_cleanup for text wrangling
    it sends a flag if there is a grouping of tweets into a block
    this allows a calling __iter__ method to aggregate text, for example to do doc2vec,
    as opposed to default setup for word2vec
    :param prefix: denotes grouping, can use 'all' as a default
    :param twl: list of dict of tweets to stream
    :return NO return value for generator Fx's
    """
    agg_wrds: int = 0
    if not prefix:
        prefix: str = "all"
    for x in twl:
        linex: str = x['text']
        if len(linex) > 10:
            wtok, agg_wrds = text_cleanup(linex, agg_wrds)
            yield wtok, prefix

    print(f"    ---- streamed {agg_wrds} words from {len(twl)} tweets  ----")

    return

class TweetyBird:
    """
    LyricLou is a class that generates sequences of words for a word2vec training algorithm
    """
    topics: list = ['workplace', 'technology', 'personaliztion']

    def __init__(self, ds: list, grp: str='workplace'):
        if grp:
            if grp == "all":
                self.topic: list = TweetyBird.topics
            elif isinstance(grp, str):
                if grp in TweetyBird.topics:
                    self.topic: str = grp
                else:
                    print(f"TweetyBird says...{grp} is not a valid group")
                    raise Exception
            elif isinstance(grp, list):
                self.topic: list = []
                for grpx in grp:
                    if grpx in TweetyBird.genres:
                        self.topic.append(grpx)
                    else:
                        print(f"TweetyBird says... {grpx} is not a valid group")
        self.dset: list = ds
        self.corpbow = []
        self.corpdic: Dictionary = {}
        self.word_count = 0
        self.word_freq: dict = {}
        self.users: list = []
        self.stream_by_trak: bool = False
        self.calc_word_freq()

    def __iter__(self):
        docsents: list = []
        currentblk: str = ""
        word_ct: int = 0
        stream_ct: int = 0
        streamed_tws: int = 0
        for lyrs, blck in feed_tweets(prefix=self.topic, twl=self.dset):
            if blck and currentblk != blck:
                currentblk = blck
            if lyrs:
                word_ct += len(lyrs)
                docsents.extend(lyrs)
            if word_ct >= 5000:
                # send list of list of word tokens
                yield docsents
                streamed_tws += len(docsents)
                stream_ct += word_ct
                word_ct = 0
                docsents = []

        self.word_count = stream_ct
        print(f"    TweetyBird streamed {stream_ct} words")

        return

    def __len__(self):
        if not self.word_count:
            self.word_count = self.get_word_count()
        return self.word_count

    def create_bow(self):
        """
        create a gensim corpora.Dictionary from the corpus for this object,
        then create Bag of Words corpus from the Dictionary
        persist the Dictionary and BoW as instance variables
        :return:
        """
        if self.corpdic:
            print("  Dictionary and BoW already calculated")
            print(f"    Dictionary Bag of Words is {len(self.corpbow)} words long")
        else:
            if self.word_count != 0:
                # need word frequencies if we ain't got 'em
                self.calc_word_freq()
        blck_corpus: list = []
        artist_lines: list = []
        total_wrds: int = 0
        cur_blck: str = ""
        for lyrs, blck in feed_tweets(self.topic, self.dset):
            if not cur_blck.startswith(blck):
                if not cur_blck == "":
                    blck_corpus.append(blck_lines)
                blck_lines = []
                cur_blck = blck
                self.users.append(cur_blck)

            # aggregate lines for each artest, then append to lyric_corpus
            blck_lines.extend(lyrs)
            total_wrds += len(lyrs)

        # existence of word in word_freq ensures freq of at least 2
        lyriclines = [[wrd for wrd in line if self.word_freq.get(wrd)] for line in blck_corpus]

        self.corpdic = Dictionary(lyriclines)
        self.corpbow = [self.corpdic.doc2bow(line) for line in blck_corpus]
        print("create_bow complete: %d sentences" % len(self.corpbow))

        return

    def calc_word_freq(self, min_count: int = 2):
        """
        create a dictionary of words and their frequency, configured to be called once
        also parses any words with a count of less than 2
        :param min_count: only include words that occur at least this many times
        :return:
        """
        if self.word_count != 0:
            print("calc_word_freq: already has %d words" % self.word_count)
        else:

            for lyrs, blck in feed_tweets(prefix=self.topic, twl=self.dset):
                # tok_line: list = [self.lemma.lemmatize(w) for w in lyrs]
                for wrd in lyrs:
                    if wrd not in self.word_freq:
                        # unique (new) words...
                        self.word_count += 1
                        self.word_freq[wrd] = 1
                    else:
                        # repeat words...
                        self.word_freq[wrd] += 1

            self.word_freq = {k: v for k, v in self.word_freq.items() if not v < min_count}
            self.word_freq = {k: v for k, v in sorted(self.word_freq.items(),
                                                      key=lambda item: item[1], reverse=True)}
        print(f"calc_word_freq: {len(self.word_freq)} words with min_count {min_count}")

        return

    def get_word_count(self):
        if self.dset:
            ctx: int = 0
            if isinstance(self.dset, list):
                for tw in self.dset:
                    ctx += len(tw['text'].split())

            return ctx
