import os
import nltk
import re
import pickle
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

CORPUS_DIR = os.path.join(os.path.dirname(__file__),'nltk_data')

if CORPUS_DIR not in nltk.data.path:
    nltk.data.path.insert(0,CORPUS_DIR)

tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)
porter = PorterStemmer()

CORPORA = ['stopwords','twitter_samples']

def ensure_corpora_downloaded():
    for corpus in CORPORA:
        try:
            nltk.data.find(f'corpora/{corpus}')
        except LookupError:
            nltk.download(corpus,download_dir=CORPUS_DIR,quiet=True)

def process_tweet(tweet):
    tweet = re.sub(r'^RT[\s]+','',tweet)
    tweet = re.sub(r'https?://[^\s\n\r]','',tweet)
    tweet = re.sub(r'#','',tweet)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweet_clean = []
    for word in tweet_tokens:
        if (word not in stopwords and word not in punctuation):
            tweet_clean.append(word)
    tweet_tokens.clear()
    for word in tweet_clean:
        stem_word = porter.stem(word)
        tweet_tokens.append(stem_word)

    return tweet_tokens

def build_freqs(tweets,ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for tweet,y in zip(tweets,yslist):
        for word in process_tweet(tweet):
            pair = (word,y)
            freqs[pair] = freqs.get(pair,0) + 1
    return freqs

def create_training(train_set, freqs:dict):
    train_array = np.zeros((len(train_set),3))
    for i,tweet in enumerate(train_array):
        train_array[i,0] = 1
        pos_con = 0
        neg_con = 0
        for word in process_tweet(tweet):
            pos_con += freqs.get((word,1),0)
            neg_con += freqs.get((word,0),0)
        train_array[i,1] = pos_con
        train_array[i,2] = neg_con

