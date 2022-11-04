# ** gs_tweet - Python 3.9 application to create, scrub, and mine datasets of tweets **
gs_tweet integrates with Postman to make Twitter API calls and to read and parse JSON responses received via Postman

This application has two control scripts:
## ** gs_main.py **
    - run this to read JSON files of Twitter tweets and users, scrub and enrich the dataset, and finally save to file.
    - After the import statements, there is a group of boolean variables used to control what sections of the script are run.
      Examples are display_dates, creat_dataframes, run_sentiment, or run_word_tokenize. 
      
    - Performs a series of scrubbing and formatting tasks.  These functions use controlling parameters and read parameters of lists and sets 
      that allow customizing this app to whatever dataset of tweets is desired.
        - Example: gs_data_dict.py is 'data dictionary' of constant definitions used by the app.  Two sets stored here are GOOD_HASHES and
          ANTI-HASHES, which can be passed to filters to pass stream of Tweets through negative match culling and/or positive match pass-through.

    - In the 'save_dataset' section of the gs_main script, a scrubbed, enriched, formatted dataset and supplementary structures like 
      word tokens and hashtag dicts, can be saved to JSON format files.  This allows clean datasets to be read in to do the fun stuff, the 
      analytics or visualization, without repeating initial processing.   
    
    - Currently this script also has a 'do_vectors' section to instantiate a custom class for tweets that includes a generator-iterator, which allows
      it to be used to train word vectors using gensim.  full disclosure- this part is drafty, needs to be expanded on and moved to the 2nd stage
      'main_postsave' script, where I read in already-cleaned datasets to do analytics or visualizations.

## ** main_postsave.py **
   - this script reads in scrubbed, validated Tweet datasets and does analysis and visualization tasks.  It is more like a separate
     application- one that simply assumes it is reading in clean, scrubbed datasets, and thus doesn't have to load and run all the stuff from gs_main.
    
## ** gs_data_dict.py **  - internal data dictionary and constant definitions

There are good reasons to have an internal data model that deviates from the Twitter data dictionary.  For one thing, twitter currently passes
responses in 'plain' 1.1, 'Native-Enriched', and 'V2' formats depending on the request endpoint and the access level (standard, elevated-sandbox,
Premium, Enterprise) of the authenticated user making the API call.  And of course, there are attributes (or features) that I derive from the 
Twitter response fields, such as summing the Twitter metrics counts for Quote Tweets, Retweets, and Replies into an aggregate field I call
'influence'.

So...
In gs_data_dict.py, I have a dict called TWEET_RECORD that contains field names and data types for my internal.   As one of the final calls in my
scrubbing and enriching pipeline, I call 'cleanup_tweet_records' which reads this data dict and validates fields and data types for my dataset.
This isn't any big technical accomplishment :-), I'm just mentioning it as I find it a helpful, standard practice to use to keep data well
structured and make sure types and values are always as expected!

There are some functions that definitely need refactoring and need to be split into more atomic, cohesive components.
A TODO I am still thinking about is to track metadata for a dataset as is passes through
parsing, formatting, scrubbing, enriching, and validating.  
I'd like to apply a set of discrete codes as metadata to indicate dataset state, then by knowing what state I need to feed a task like
sentiment scoring, or Vector Training, I might get to a point where the scrubbing can be controlled with automation.  
Just an idea for now, anything to cut the manual effort with data cleansing!
