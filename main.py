import pandas as pd
import numpy as np
import string
import re
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
from utils import emo
import emoji
import string
from wordcloud import WordCloud
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from modules.bundling import *
from modules.preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer


pos_emo = ['admiration' , 'amusement' , 'approval' , 'caring' , 'curiosity' , 'desire' , 'excitement' , 'gratitude' , 'joy' , 'love' , 'optimism' , 'pride' , 'realization' ,'relief' ,'surprise' ,'neutral']
neg_emo = ['anger' , 'annoyance' , 'confusion' , 'disappointment' , 'disgust' , 'embarrassment' , 'fear' , 'grief' , 'nervousness' , 'remorse' , 'sadness' , '']


df = bundle_data(pos_emo,neg_emo)
df['text'] = df['text'].apply(lambda x: convert_emojis(x)) # converting emojis to text (Time consuming step !!!)
df['text'] = df['text'].apply(lambda x: convert_emoticons(x)) # converting emoticons to text
df['text'] = df['text'].apply(lambda x: x.lower())   # converting data to lowecase
df['text'] = df['text'].apply(lambda x: decontracted(x)) 
df['text'] = df['text'].apply(lambda x: remove_urls(x))  
df['text'] = df['text'].apply(lambda x: remove_punctuation(x)) # removing unwanted characters


# saving the preprocessed data
df.to_csv('output/cleaned_text_data.csv',header=['Text', 'Positive','Negative'])


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['text'], 20)
tup = []
for word, freq in common_words:
    tup.append((word, freq))
    
    
kf = pd.DataFrame(tup,columns=['word', 'freaquency'])
kf.to_csv('output/uni-gram_with_stop_words.csv',index=False)


def get_top_n_word(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_word(df['text'], 20)
tup = []
for word, freq in common_words:
    tup.append((word, freq))
    
    
kf = pd.DataFrame(tup,columns=['word', 'freaquency'])
kf.to_csv('output/uni-gram_without_stop_words.csv',index=False)

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['text'], 20)
tup = []
for word, freq in common_words:
    tup.append((word, freq))
    
    
kf = pd.DataFrame(tup,columns=['word', 'freaquency'])
kf.to_csv('output/bi-gram_with_stop_words.csv',index=False)


def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df['text'], 20)
tup = []
for word, freq in common_words:
    tup.append((word, freq))
kf = pd.DataFrame(tup,columns=['word', 'freaquency'])
kf.to_csv('output/tri-gram_without_stop_words.csv',index=False)


wordcloud = WordCloud(background_color="white",width=1600, height=800).generate(' '.join(df['text'].tolist()))
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.savefig('output/WordCloud.png')