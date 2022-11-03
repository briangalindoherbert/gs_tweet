# encoding=utf-8
"""
tweet_data_dict contains CONSTANTS and word lists for nlp processing.
tw2vec is a pretrained vector model trained on a tweet corpus from google

ADDITIONAL SUB-DIRECTORIES OFF GSTWEET FOR THIS PROJECT:
./project/  - articles and documents on this topic
./twitter/  - files with batches of tweets, json format, from twitter developer api
    endpoints which I access using Postman.  easier to tweek queries and check results
    than via scripting the http get in python.
/templates/ - html, javascript, json and yaml templates.  html+js as I'm looking at some
    cool d3 viz stuff I can do with this data, json for playing with parsing schemas, and
    yaml for config files if I make the twitter api calls from py scripts.
/output/ - my 'deliverables' such as serializing my data to file, the gensim models I
    generate, wordclouds, and other visualizations and saved data.
/models/ - pretrained or pre-labeled data for word2vec or nltk models, such as large
    vocabulary files with vector embeddings or tweets or phrases with sentiment labels
"""

import datetime as dt

MODELDIR = '/Users/bgh/dev/NLP/models/'
RAWDIR = '/Users/bgh/dev/pydev/gs_labor/rawdata/'
TW_DIR = '/Users/bgh/dev/pydev/gs_labor/tweets/'
OUTDIR = '/Users/bgh/dev/pydev/gs_labor/output/'
W2VEC_PRE = '/Users/bgh/dev/pydev/NLP/models/freebase-vectors-skipgram1000-en.bin'
TW2VEC_PRE = '/Users/bgh/dev/pydev/NLP/models/word2vec_twitter_tokens.bin'

USER_RATING: dict = {'1276297503931904000': 5, }
BAD_IDS: set = {'1525587002728411136', '1537132116562939905', '1567164270286094338', '1525856745867857921',
                '1567559513649061896', '1568760360190251008', '1564942020447182849', '1564440488744292352',
                '1525607804450521092', '1541564317333704706', '1574827842797391872', '1573369460366675969',
                '1572631039423709184', '1549783381038219265', '1549778576190238720', '1575834447223685121',
                '1575366076455882752', '1571891865691254784', '1550493219842101248', '1571891865691254784',

}
TWEET_RECORD: dict = {'id': str, 'text': str, 'rawtext': str, 'conv': str, 'qrr': int, 'quot': int,
                      'rtwt': int, 'rply': int, 'fave': int, 'missing_from': str,
                      'sent': dt.datetime, 'tdate': str, 'ttime': str, 'uname': str,
                      'userid': str, 'ufollow': int, 'ufriends': int, 'hashes': list,
                      'mentions': list, 'urls': list, 'compound': float,
                      'neu': float, 'pos': float, 'neg': float, 'lang': str,
                      'rply_id': str, 'rply_uid': str, 'rply_uname': str,
                      'rt_id': str, 'rt_src_dt': dt.datetime,
                      'rt_qrr': int, 'rt_quot': int, 'rt_rtwt': int, 'rt_rply': int,
                      'rt_fave': int, 'rt_uname': str, 'rt_ufollow': str, 'rt_ufriends': str,
                      'qt_id': str, 'qt_src_dt': dt.datetime, 'qt_text': str,
                      'qt_qrr': int, 'qt_quot': int, 'qt_rtwt': int, 'qt_rply': int,
                      'qt_fave': int, 'qt_uid': str, 'qt_uname': str, 'qt_ufollow': str,
                      'qt_ufriends': str, 'qt_compound': float, 'qt_neu': float,
                      'qt_neg': float, 'qt_pos': float, 'wrdscore': int,
                      }
ANALYTICS_FIELDS: dict = {'wrdscore': {'dtype': int,
                          'def': "applies tfidf word scores to calculate information value of tweet"},
                          'distribution': {'dtype': int,
                          'def': "if qt, rt or reply status, sum of log of reply and quote counts"},
                          'urating': {'dtype': int, 'def': '0-9 rating of tweets by this user'}
                         }
GOOD_HASHES: set = {'quietquitting', 'worklifebalance', 'employeeengagement', 'workplace',
                    'futureofwork', 'mentalhealth', 'work', 'burnout', 'careers',
                    'business', 'companyculture', 'culture', 'digitaltransformation',
                    'employeeexperience', 'employees', 'employment', 'entrepreneur',
                    'flexibileschedules', 'greatresignation', 'humanresources', 'leadership',
                    'innovation', 'hybridwork', 'inspiration', 'management', 'motivation',
                    }
BAD_HASHES: set = {'onlineradio', 'mlb', 'cardinals', 'dubuque', 'yakshagana', 'blackforest',
                   'naturelovers', 'privateeye', 'crescentwarrior', 'bhubaneswar', 'newmexico',
                   'travelmore', 'newmexico', 'islam', 'borisjohnson', 'vladmirputin', 'nowhiring',
                   'emmyawards', 'scubadivewithjah', 'nowplaying', 'booking', 'christiclarity',
                   'xinjiang', 'china', 'nagaland', 'pakistan', 'grupomusical', 'sb19',
                   'mysteryskullsanimated', 'gujarat', 'delhi', 'translation', 'shabirahluwalia',
                   'radhamohan', 'blockbustersvp', 'vietnam'
}
TOPIC_WORDS: set = {'application', 'assets', 'balance', 'build products', 'burning out',
                    'burnout', 'business', 'candidate', 'clock', 'collaboration', 'colleague',
                    'cv', 'company', 'companies', 'corporate', 'culture', 'employee', 'employer',
                    'employment', 'engagement', 'ethic', 'experience', 'fired', 'gen z',
                    'generation', 'genx', 'genz', 'hardwork', 'hired', 'home', 'hostile work',
                    'gen x', 'human', 'implement', 'improve', 'innovate', 'innovative', 'input',
                    'job', 'labor', 'love', 'manage', 'management', 'mental health', 'millenials',
                    'money', 'organization', 'project team', 'quietquitting', 'quit', 'quitting',
                    'rat race', 'remote', 'resign', 'resignation', 'resume', 'resources',
                    'shareholder', 'staff', 'systems', 'team', 'time', 'toxic',
                    'transition', 'week', 'work', 'workforce', 'worklife', 'workplace',
                    'unemployment', 'professional', 'incentive'
                    }
ANTI_TOPICS: set = {'trumpism', 'trump', 'biden', 'current openings', 'current jobs', 'GOP',
                    'venezuela', 'gujarat', 'kenya', 'election', 'mueller', 'comey', 'for sale',
                    'commercialfreeradio', 'travelmore', 'spendless', 'delhi','vietnam', 'china',
                    'mlb', 'privateeye', 'travelmore', 'nowplaying',
                    }
OFFTOPIC: list = ['writingskills', 'comey', 'biden', 'trump', 'learnenglish', 'remotelearning',
                  'videoshorts', 'mueller', 'venezuela', 'kenya', 'kenyan', 'gujarat',
                  'taxpayer', 'government', 'tories', 'taxes', 'democrat', 'republican'
                  ]
STOP_CLOUD = ["trump", "biden", "election", "seonghwa", "shib", "enugu",
                      "gurugram", "fajita", "arteta", "pylades", "viviyukino",
                      "gujaratelection", "rick", "euros", "sargon", "lanka"]

GS_ABSOLUTE = ["always", "horrible", "never", "perfect", "worthless", "useless",
               "infinitely", "absolutely", "completely", "totally", "exponentially",
               "idiotic"]
GS_EXTREME = ["insane", "evil", "psycho", "idiot", "rube", "crazy", "neurotic", "retarded",
              "stupid"]
GS_BALANCE: list = ["relative", "preferable", "optimal", "better", "inferior", "superior"]
# these are standard Adverb and Pronoun STOPS on many nlp projects
GS_ADVB = ["am", "are", "do", "does", "doing", "did", "is", "was", "were"]
GS_PPRON = ["we", "he", "her", "him", "me", "she", "them", "us", "they"]
# contractions expansions, includes forms with missing apostrophe
GS_CONTRACT = {
    "-": " ",
    "ain't": "aint",
    "aren't": "are not",
    "arent": "are not",
    "can't": "can not",
    "cant": "can not",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "dont": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "isnt": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it'll've": "it will have",
    "its": "it is",
    "it's": "it is",            # the contraction is often mis-spelled
    "let's": "let us",
    "ma'am": "mam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "oclock",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "theyll": "they will",
    "they're": "they are",
    "theyre": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "whats": "what is",
    "what've": "what have",
    "when's": "when is",
    "whens": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "whos ": "who is ",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "wont": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "yall",
    "you'd": "you would",
    "youd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}
# Words to Remove: standard plus stops based on project type, context, or tfidf analysis
GS_STOP = ["a", "about", "actual", "actually", "almost", "also", "among", "an", "and",
           "already", "any", "approximately", "around", "as", "at", "back", "basically",
           "because", "but", "by", "cause", "could", "dear", "else",
           "ever", "even", "for", "from", "happen", "him", "his", "her", "hers",
           "how", "however", "if", "in", "into",
           "it", "its", "just", "least", "let", "lets", "likely", "many",
           "may", "me", "might", "most", "must", "much", "my", "now", "of", "often",
           "onto", "or", "other", "rather", "really", "seems", "should",
           "simply", "since", "so", "some", "something", "sometimes", 'still', 'such',
           "than", "that", "the", "their", "them", "then", "there", "these", "they",
           "theyre", "thinks", "this", "those", "though", "thus", "to", "too", "well",
           "while", 'why', "will", "with", "yet", "your", 'youre'
           ]
STOP_TWEET = ['RT', "rt", '…', '_', '(', ')', '[', ']', '__', ':', '"', '️', '"',
              '/', ']', '|', '[', ']', 'acdvd', 'additional', 'de', 'A', 'le',
              'affect', 'again', 'ahead', 'ake', 'als', 'anybody', 'anyway', 'apostrophe_',
              'app', 'asterisk', 'bara', 'being', 'besides', 'breaka', 'chuchilips', 'cian', 'clearly',
              'colon', 'comma', 'cos', 'definitely', 'delve', 'despite',
              'differen', 'doing', 'dr', 'each', 'erstes', 'everyone', 'fairly',
              'flies', 'fully', 'going', 'got', 'guess', 'hashtag',
              'having', 'hea', 'here', 'hey', 'hows', 'hyphen', 'id',
              'ill', 'ings', 'ins', 'instead', 'ipad', 'iphone', 'ipod', 'ive',
              'keeping', 'literally', 'lot', 'maybe', 'micr',
              'name', 'nearly', 'nevertheless', 'notes', 'orbn',
              'possibly', 'potentially', 'put', 'quite', 'quotation', 'remains',
              'seem', 'semicolon', 'single', 'slash', 'soo',
              'supe', "talk", "look", 'tha', 'thats', 'themselves', 'theres', "'save",
              'tho', 'thst', 'trying', 'type', 'underscore',
              'uns', 'until', 'vary', 'way', 'whe', 'whether', 'which', 'ya', 'yep', 'yer', 'youd']
STOP_PREP = ["a", "ah", "about", "also", "among", "am", "an", "another", "and", "any", "are", "as",
             "at", "be", "because", "but", "by", "can", "cause", "could", "dit", "do", "doo",
             "either", "else", "ever", "for", "from", "get", "got", "ha", "hah", "have", "he",
             "hey", "how", "however", "huh", "if", "is", "it", "just", "la", "let", "lot",
             "like", "likely", "may", "me", "might", "must", "na", "nah", "not", "of", "off",
             "often", "oh", "on", "ooh", "ooooh", "or", "so", "some", "than", "that",
             "the", "their", "then", "these", "they", "this", "tis",
             "to", "too", "twas", "uh", "us", "was", "way", "what", "when", "which",
             "while", "whoa", "whom", "will", "with", "would", "yet", "ya", "yo"
             ]
# bracket special chars for RE compares. RE and python compare (if x in Y) different
NOT_ALPHA = r"(?i)[^a-z\s]"
JUNC_PUNC = "[*+%:&;/',]"
XTRA_PUNC = r"[[:punct:]]"
END_PUNC = "[.!?]"      # keep ! and ?, they both have effect on tweet sentiment
GS_SENT_END: list = ["!", "?", ";", ".", "..", "..."]
# capture special Tweet text: user_mention, hashtag, urls, stuff inside paras, punctuation
GS_PAREN = "\((.+?)\)"
GS_URL = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'\
         r'[-a-zA-Z0-9()@:%_\+.~#?&//=]*'
GS_MENT = "@(\w*)"                      # find user mentions as in '@UserName'
GS_HASH = "[#](\w*)"
GS_UCS4 = r"\\\\[x][0-9,a-f]{4}"        # find ucs-2 aka double-byte characters
GS_UCS = "\\u[0-9,a-f]{4}"
# GS_UCS2 shows ucs-1 symbol equivalent to ucs-2 if symbol exists
GS_UCS2: dict = {"\u003b":  ";",
                 "\u003c":  "<",
                 "\u003e":  ">",
                 r"\u003f":  r"?",
                 r"\u0040":  r"@",
                 r"\u00a1":  r"!",       # '¡'
                 r"\u00a2":  "",         # '¢'
                 r"\u00a3":  "brP",      # '£'
                 r"\u00a4":  "",         # '¤'
                 r"\u00a6":  r":",        # '¦'
                 r"\u00a8":  "",             # unlaut  '¨'
                 r"\u00a9":  "cpyrt",        # '©'
                 r"\u00ae":  "reg copyrt",     # reg copyrt  '®'
                 r"\u00b6": r"<p>",          # para mark '¶'
                 r"\u00b8": r".",           # period "."
                 r"\u00bd":  "1/2",          # symbol '½'
                 r"\u00bf":  "",             # spanish inverted question  '¿'
                 r"\u00e0":  "a",            # a with accent grave  'à'
                 r"\u00e7":  "c",            # c with lower accent   "ç"
                 r"\u2012":  "-",
                 r"\u2013":  "–",
                 r"\u2014":  "–",
                 r"\u2015":  "–",
                 r"\u2016":  "",          # '‖'
                 r"\u2017":  "",          # '‗'
                 r"\u2018": r"'",
                 r"\u2019": r"'",
                 r"\u201a": r",",
                 r"\u201b": r"'",
                 r"\u201c": r"'",
                 r"\u201d": r"'",
                 r"\u201e": r"'",
                 r"\u201f": r"'",
                 }
UCS_SYM = "\\[u][0,2]{2}[0-9,a-f]{2}"
# lists of multi-char symbols, utf_meaning allows us to translate approx. emotion
STOP_NONALPHA: list = ['🚨', '⚽', '🏻', '🔴', '👇', '🚫', '😭', '🔵', '✅', '❌',
                       '💥', '🏽', '🗣', '💔', '🙄', '💪', '🙌', '🤷', '🙏', '👉',
                       '🏆', '🏉', '•', '👌', '🟡', '😒', '💛', '💰', '🪦', '😬',
                       '✊', '🥳', '🍭', '🌶', '😜', '⚡', '🐍', '🤡', '🎉', '📻',
                       '🎙', '💙', '🤦', '❗', '😤', '🔥', '👀', '😍', '😳', '😎',
                       '🏾', '😅', '💀', '💸', '✍', '💯', '🤝', '💵', '🤬', '😀',
                       '😇', '😌', '📺', '🚀', '😹', '💩', '💬', '🧐', '🟢', '🐍', '⚧',
                       '🤓', '👊', '😔', '🤮', '🎩', '😡', '😩', '🤨', '😐', '🤩',
                       '🤞', '❓', '💭', '🆚', '👻', '🥴', '♦', '🖊', '♥', '🥇', '🥰',
                       '🔰', '❔', '🌍', '🤐', '😏', '🤯', '👑', '🧘', '📕', '📸', '🖤',
                       '🤑', '⛔', '📰', '⏰', '✨', '🥱', '🟠', '🌎', '⏲', '📋', '🏟',
                       '📊', '😘', '◾', '🕒', '💻', '📝', '🪓', '🌱', '😱', '📌', '👥',
                       '🌈', '🎥', '🎫', '🎮', '🤒', '⏳', '📍', '🖥', '🌏', '▔', '😲',
                       '👈', '👚', '👖', '⬅', '🐏', '🥉', '🔹', '🏿', '💶', '📢',
                       '🏐', '💷', '💚', '🌐', '🥧', '🧗', '🤰', '😪', '😵', '🗑', '🥈',
                       '🌙', '📩', '🎧', '🌿', '🍃', '🦊', '🏳', '🍿', '🦁', '🦋', '🗳',
                       '🍺', '🧡', '😰', '🦆', '📹', '🗞', '🔔', '💹', '🦈', '🐦', '🐍',
                       '🔗', '🙈', '📽', '📱', '💴', '🤧', '⭐', '🥀', '🤭', '🔞', '🤖',
                       '😴', '📦', '🌚', '🎖', '🤲', '🤫', '🏹', '🧢', '😑', '🗯', '🍓',
                       '🤍', '📣', '🍻', '🕺', '🥂', '🍾', '🎂', '🎊', '🌹', '🎁', '🏃',
                       '🌑', '🛑', '🐺', '🧵', '🍑', '🔲', '🌑', '🌹', '🍑', '🍓', '🍻',
                       '🐦', '🐺', '💹', '📣', '🔲', '🕺', '🛑', '🤍', '🥂', '🥵', '🦈',
                       '🧐', '🧵', '🥂', '🏃', '💹', '5️⃣0️⃣', '🕺', '🐦', '📸', '🛑',
                       '⚧', '⏪' '🔗', '💷', '🍀', '🥅', '🍾', '🎁', '🎂', '🎊', '🏃',
                       '🔽', '⏩', '🔁', '🔄', '⏩', '🔟', '🥵', '🤍', '🎊', '🌹', '🎁',
                       '🍾', '🎂', '📣', '🍻', ':p', '🐺', '🧵', '🍑', '🦈', '🪐', '🌑',
                       '🌙', '🍭', '🔊', '🏼', '🪐', '🎈', '🚩', '🟥', '📃', '🥵', '🔲',
                       '⌚', '⏩', '┏', '┓', '┳', '┻', '╭', '╮', '▉', '▏', '░', '▕',
                        '⚡', '⚧', '➡', '🇨', '⚠', '🇦', '🇳', '🇫', '🇩', '«', '»',
                        '⁦', '⁩', '🇮', '⬇', '🇹', '🇸', '🇺', '⚪', '🇪',
                        '♂', '❤', '…', '€', 'ℹ️', '⚒', '🎶', '🏴', '⌚',
                        '╭', '╮', '▏', '▕', '┏', '┳', '┓', '┻', '░', '⇢', '▉', ]
UTF_MEANING: dict = {'😋': ' it is very good ',
                     '🙂': ' it is good ',
                     '🤣': ' this is hilarious ',
                     '😂': ' this is funny ',
                     '🤔': ' this is curious ',
                     '🤗': ' this is wonderful ',
                     '👍': ' this is good ',
                     '🤙': ' everything is cool ',
                     '👋': ' I applaud this ',
                     '👎': ' I reject this ',
                     '😫': ' this is horrible ',
                     '😆': ' had a good laugh ',
                     '🥲': ' this is sad ',
                     '😊': ' good stuff ',
                     '😞': ' sad situation ',
                     '😁': ' makes me smile ',
                     '🤬': ' this is maddening ',
                     '😮': ' this is shocking ',
                     '🚫': ' prohibit this ',
                     '🖕': ' go to hell ',
                     '😠': ' frustrated ',
}
# repr(xx).strip("'") displays char represented by \uxxxx code
GS_EMOJI: dict = {"\ud83d\udea8": "🚨",
                  "\ud83e\udd23": "🤣",
                  "\u26aa\ufe0f": "⚪",
                  "\u26a0\ufe0f": "⚠",
                  "\u26BD\uFE0F": "⚽️",
                  "\u2b07\ufe0f": "⬇",
                  "\ud83e\udd2c": "🤬",  # angry, cussing head
                  "\ud83d\udcca": "📊",
                  "\ud83d\udde3\ufe0f": "🗣",
                  "\ud83d\udeab": "🚫",
                  "\ud83c\uddea\ud83c\uddfa": "🇪🇺",
                  "\ud83c\udde9\ud83c\uddea": "🇩🇪",
                  "\ud83d\ude4c": "🙌 ",
                  "\ud83d\udd34\u26aa\ufe0f": "🔴⚪",
                  "\ud83d\udd34": "🔴 ",
                  "\ud83d\udeab\ud83d\udd35": "🚫🔵",
                  "\ud83e\udd21": "🤡",
                  "\ud83d\udc80": "💀",
                  "\ud83d\udc51": "👑"
                  }
emoji_dict: dict = {
                    ":-)": "basic smiley",
                    ":)": "midget smiley",
                    ",-)": "winking smiley",
                    "(-:": "left hand smiley",
                    "(:-)": "big face smiley",
                    ":-(": "sad face",
                    ":-(-": "very sad face",
                    "8-O": "omg face",
                    "B-)": "smiley with glasses",
                    ":-)>": "bearded smiley",
                    "'-)": "winking smiley",
                    ":-#": "my lips are scaled",
                    ":-*": "kiss",
                    ":-/": "skeptical smiley",
                    ":->": "sarcastic smiley",
                    ":-@": "screaming smiley",
                    ":-V": "shouting smiley",
                    ":-X": "a big wet kiss",
                    ":-\\": "undecided smiley",
                    ":-]": "smiley blockhead",
                    ";-(-": "crying sad face",
                    ">;->": "lewd remark",
                    ";^)": "smirking smiley",
                    "%-)": "too many screens",
                    "):-(-": "nordic smiley",
                    ":-&": "tongue tied",
                    ":-O": "talkaktive smiley",
                    "+:-)": "priest smiley",
                    "O:-)": "angel smiley",
                    ":-<:": "walrus smiley",
                    ":-E": "bucktoothed vampire",
                    ":-Q": "smoking smiley",
                    ":-}X": "bowtie smiley",
                    ":-[": "vampire smiley",
                    ":-{-": "mustache smiley",
                    ":-{}": "smiley wears lipstick",
                    ":^)": "smiley with personality",
                    "<:-l": "dunce smiley",
                    ":=)": "orangutan smiley",
                    ">:->": "devilish smiley",
                    ">:-l": "klingon smiley",
                    "@:-)": "smiley wearing turban",
                    "@:-}": "smiley with hairdo",
                    "C=:-)": "chef smiley",
                    "X:-)": "smiley with propeller beanie",
                    "[:-)": "smiley with earbuds",
                    "[:]": "robot smiley",
                    "{:-)": "smiley wears toupee",
                    "l^o": "hepcat smiley",
                    "}:^)": "pointy nosed smiley",
                    "(:-(": "saddest smiley",
                    ":-(=)": "bucktooth smiley",
                    "O-)": "message from cyclops",
                    ":-3": "handlebar mustache smiley",
                    ":-=": "beaver smiley",
                    "P-(": "pirate smiley",
                    "?-(": "black eye",
                    "d:-)": "baseball smiley",
                    ":8)": "piggy smiley",
                    ":-7": "smirking smiley",
                    "):-)": "impish smiley",
                    ":/\\)": "bignose smiley",
                    ":-(*)": "vomit face",
                    ":(-": "turtle smiley",
                    ":,(": "crying smiley",
                    ":-S": "confuzled face",
                    ":-[ ": "unsmiley blockhead",
                    ":-C": "real unhappy smiley",
                    ":-t": "pouting smiley",
                    ":-W": "forked tongue",
                    "X-(": "brain dead"
}

IDIOM_MODS = {'dead end': -2.5, 'male privilege': -1.5, "good guys": 0.5}
SPECIAL_CASE_IDIOMS = {'bad ass': 1.5, 'cut the mustard': 2, 'hand to mouth': -2,
                       'kiss of death': -1.5, 'the bomb': 3, 'the shit': 3, 'yeah right': -2}
VADER_MODS = {"hugs": 1.0, "sociopathic": -2.5, "cartel": -1.0, "paycheck": -0.5,
              "blunder": -0.5, "socialism": -1.5}

QUOTEDASH_TABLE = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–－᠆–-", "'''''-----")])
CHAR_CONVERSION = {
    u"\u200B": "",  # this one's a bugger- 'zero-length space' unicode- aka invisible!
    u"\u2002": " ",
    u"\u2003": " ",
    u"\u2004": " ",
    u"\u2005": " ",
    u"\u2006": " ",
    u"\u2010": "-",
    u"\u2011": "-",
    u"\u2012": "-",
    u"\u2013": "-",
    u"\u2014": "-",
    u"\u2015": "-",
    u"\u2018": "'",
    u"\u2019": "'",
    u"\u201a": "'",
    u"\u201b": "'",
    u"\u201c": "'",
    u"\u201d": "'",
    u"\u201e": "'",
    u"\u201f": "'",
    u"\u2026": "'",
    "－": "-",
    u"\u00f6": "o",         # this and next inspired by Motley Crue
    u"\u00fc": "u",
}

TRACE_COLRS = ["rgb(255, 0, 51)", "rgb(204, 102, 51)", "rgb(102, 102, 102)",
               "rgb(0, 102, 102)", "rgb(51, 204, 51)", "rgb(0, 204, 102)",
               "rgb(255, 51, 153)", "rgb(255, 102, 204)", "rgb(51, 51, 51)",
               "rgb(102, 0, 153)", "rgb(0, 102, 153)", "rgb(51, 102, 153)",
               "rgb(204, 102, 51)", "rgb(153, 102, 0)", "rgb(153, 204, 153)",
               "rgb(204, 204, 204)", "rgb( 102, 51, 51)", "rgb(51, 102, 153)",
               "rgb(153, 51, 102)", "rgb(0, 204, 102)", "rgb(255, 102, 51)",
               "rgb(204, 204, 102)"]
GSC = {
    "dkblu": "rgb(0, 102, 153)",
    "ltblu": "rgb(0, 153, 255)",
    "grn": "rgb(0, 204, 102)",
    "oblk": "rgb(51, 51, 51)",
    "prpl": "rgb(51, 51, 153)",
    "dkgrn": "rgb(51, 102, 51)",
    "dkryl": "rgb(51, 102, 153)",
    "brwn": "rgb( 102, 51, 51)",
    "brgry": "rgb( 102, 102, 102)",
    "drkrd": "rgb(153, 51, 102)",
    "brnz": "rgb(153, 102, 0)",
    "gray": "rgb(153, 153, 153)",
    "brnorg": "rgb(153, 102, 51)",
    "lgrn": "rgb(153, 153, 51)",
    "slvr": "rgb(153, 153, 153)",
    "org": "rgb(204, 102, 51)",
    "gld": "rgb(204, 153, 51)",
    "olv": "rgb(204, 204, 102)",
    "beig": "rgb(204, 204, 153)",
    "ltgry": "rgb(204, 204, 204)",
    "mgnta": "rgb(255, 51, 255)",
    "owht": "rgb(204, 255, 204)",
    "white": "rgb(255, 255, 255)",
}
