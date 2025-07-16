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

def extract_features(tweet, freqs:dict):
    X = np.zeros(3)
    word_l = process_tweet(tweet)
    X[0] = 1
    pos_con = 0
    neg_con = 0
    for word in word_l:
        pos_con += freqs.get((word,1),0)
        neg_con += freqs.get((word,0),0)

    X[1] = pos_con
    X[2] = neg_con

def create_training_matrix(train_set,freqs):
    train_X = np.zeros((len(train_set),3))
    for i,tweet in enumerate(train_set):
        train_X[i,:] = extract_features(tweet,freqs)
    return train_X

#TRAINING UTILS
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def gradient_descent(train_X,train_y,theta,alpha,num_iters):
    m = train_X.shape[0]
    history_loss = []
    history_theta = []

    for i in range(num_iters):
        z = np.dot(train_X,theta)
        h = sigmoid(z)
        J = (-1/m)*(np.dot(train_y.T,np.log(h)) + np.dot(1 - train_y.T,np.log(1 - h)))
        history_loss.append(J)
        history_theta.append(theta)
        theta -= (alpha/m)*(np.dot(train_X.T,h - train_y))

    J = float(J)
    return J, theta, history_loss, history_theta

def predict_tweet_logits(tweet,freqs,theta):
    feat_tweet = extract_features(tweet,freqs)
    return sigmoid(np.dot(feat_tweet,theta))

def predict_tweet_threshold(tweet,freqs,theta,threshold = 0.5):
    feat_tweet = extract_features(tweet,freqs)
    return (1 if (sigmoid(np.dot(feat_tweet,theta)) >= threshold) else 0)

def test_theta