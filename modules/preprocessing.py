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
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords


PUNCT_TO_REMOVE = string.punctuation
UNICODE_EMO = emo.UNICODE_EMOJI
EMOTICONS = emo.EMOTICONS_EMO

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
def remove_html(text):
    return BeautifulSoup(text, "lxml").text

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def clean_tweets(x):
    x = re.sub('@\w*','',x)
    x = re.sub('\.*\.',' ',x)
    x = re.sub(' +',' ',x)
    return x

#  Converting Emoticons Start
    # In general we remove the emoticons 
    # and emoji from dataset
    # but since we are working with 
    # detecting emotions so they will
    # be replaced by their meanings

str_emo = str(EMOTICONS)[1:-1]
str_emo = str_emo.replace("\\","\\\\")
str_emo = str_emo.replace('(','\(')
str_emo = str_emo.replace(')','\)')
str_emo = str_emo.replace('[','\[')
str_emo = str_emo.replace(']','\]')
str_emo = str_emo.replace('{','\{')
str_emo = str_emo.replace('}','\}')
str_emo = str_emo.replace('.','\.')
str_emo = str_emo.replace('$','\$')
str_emo = str_emo.replace('*','\*')
ttr_emo = '{'+str_emo+'}'
mod_emo = eval(ttr_emo)

def convert_emoticons(text):
    for e,m in zip(EMOTICONS,mod_emo):
        if e in text:
            text = re.sub(m, "_".join(EMOTICONS[e].replace(",","").split()), text)
    return text

# Converting Emoticons End


def convert_emojis(text):
    return emoji.demojize(text)



def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"won’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
    return phrase


def remove_Stopwords(text ):
    stop_words = set(stopwords.words('english')) 
    words = word_tokenize( text.lower() ) 
    sentence = [w for w in words if not w in stop_words]
    return " ".join(sentence)
    
def lemmatize_text(text):
    wordlist=[]
    lemmatizer = WordNetLemmatizer() 
    sentences=sent_tokenize(text)
    for sentence in sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    return ' '.join(wordlist) 
def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr]) 
    
    return text2.lower()