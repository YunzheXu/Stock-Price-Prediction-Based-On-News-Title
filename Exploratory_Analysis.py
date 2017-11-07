# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:20:17 2017

@author: Yunzhe
"""
import csv
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import string
from nltk.corpus import stopwords

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
# In[ ]
titles = []
sentiments = []

with open("cleaned and preprocessed data from EBSCO.csv", "rb") as rf:  
    reader=csv.reader(rf, delimiter=',')
    titles = [(str(row[1])) for row in reader]
rf.close()

stop_words = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()


# define a mapping between wordnet tags and POS tags as a function
def get_wordnet_pos(pos_tag):
    
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ
    
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB
    
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN

# do sentiment by sentiwordnet
# return a dict of sentiments for each word
def get_sentiment(wordList):
    item = {}
    for word in wordList:
        s = wn.synsets(word)
        if (s != []):
            sen = str(s[0])[8:-2]       
            res = swn.senti_synset(sen)
            item.update({str(word): str(res)})
    return item

for text in titles:

    # do lemmatization
    
    # first tokenize the text
    tokens=nltk.word_tokenize(text)
    
    # then find the POS tag of each word
    # tagged_token is a list of (word, pos_tag)
    tagged_tokens= nltk.pos_tag(tokens)
    #print(tagged_tokens)

    # get lemmatized tokens
    # lemmatize every word in tagged_tokens
    le_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) \
              # tagged_tokens is a list of tuples (word, tag)
              for (word, tag) in tagged_tokens \
              # remove stop words
              if word not in stop_words and \
              # remove punctuations
              word not in string.punctuation]
    
    # get lemmatized unique tokens as vocabulary
    le_vocabulary=set(le_words)
    
    # do sentiment
    sen = get_sentiment(le_words)
    sentiments.append(sen)

# print all the sentiments for each word in each title
# PosScore and NegScore 
print (sentiments)




