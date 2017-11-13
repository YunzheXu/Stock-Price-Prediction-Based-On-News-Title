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

import matplotlib.pyplot as plt
from string import punctuation
# In[ ]
titles = []
sentiments = []
pos_words = []
neg_words = []
non_words = []

with open("1-cleaned and preprocessed data from EBSCO.csv", "rb") as rf:  
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
            if res.pos_score() > 0 and res.pos_score() > res.neg_score():
                pos_words.append(str(word))
            elif res.neg_score() > 0 and res.pos_score() < res.neg_score():
                neg_words.append(str(word))
            else:
                non_words.append(str(word))
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
#print (sentiments[0])

#print (titles)
def title_to_text(titles):
    wordsText = ''
    for row in titles:
        wordsText = wordsText + ' ' + str(row)
    return wordsText

text = ''
text = title_to_text(titles)

def count_token(text):
    token_count = {}
    tokens = text.split()
    for token in tokens:
        token = token.lower().strip().strip(punctuation)
        if token == ' ':
            break
        token_count[token] = token_count.get(token, 0) + 1    
    return token_count

def drawPic(text, n):
    
    count_per_word = count_token(text) 
    
    word_count_list=count_per_word.items()
    
    sorted_words=sorted(word_count_list, key=lambda item:-item[1])
    
    
    top_n_words = sorted_words[0:n]
    
    # split the list of tuples into two tuples
    new_words, counts=zip(*top_n_words)
    
            
    x_pos = range(len(new_words))
    
    # plot the bar chat
    plt.bar(x_pos, counts)
    
    # add the legend of each bar
    plt.xticks(x_pos, new_words)
    
    # add the label for Y-axis
    plt.ylabel('Count of Words in News Titles')

    # add title
    plt.title('Top ' + str(n) + ' Words')

    # vetically align the text of each topic
    plt.xticks(rotation=90) 
    
    # display the plot
    plt.show()    

drawPic(text, 50)

pos_text = ''
for word in pos_words:
    pos_text = pos_text + ' ' + word
    
def drawPosPic(text, n):
    count_per_word = count_token(text) 
    
    word_count_list=count_per_word.items()
    
    sorted_words=sorted(word_count_list, key=lambda item:-item[1])
    
    
    top_n_words = sorted_words[0:n]
    
    # split the list of tuples into two tuples
    new_words, counts=zip(*top_n_words)
    
            
    x_pos = range(len(new_words))
    
    # plot the bar chat
    plt.bar(x_pos, counts)
    
    # add the legend of each bar
    plt.xticks(x_pos, new_words)
    
    # add the label for Y-axis
    plt.ylabel('Count of Words in News Titles')

    # add title
    plt.title('Top ' + str(n) + ' Positive Words')

    # vetically align the text of each topic
    plt.xticks(rotation=90) 
    
    # display the plot
    plt.show()    

drawPosPic(pos_text, 50)

neg_text = ''
for word in neg_words:
    neg_text = neg_text + ' ' + word
    
def drawNegPic(text, n):
    count_per_word = count_token(text) 
    
    word_count_list=count_per_word.items()
    
    sorted_words=sorted(word_count_list, key=lambda item:-item[1])
    
    
    top_n_words = sorted_words[0:n]
    
    # split the list of tuples into two tuples
    new_words, counts=zip(*top_n_words)
    
            
    x_pos = range(len(new_words))
    
    # plot the bar chat
    plt.bar(x_pos, counts)
    
    # add the legend of each bar
    plt.xticks(x_pos, new_words)
    
    # add the label for Y-axis
    plt.ylabel('Count of Words in News Titles')

    # add title
    plt.title('Top ' + str(n) + ' Negative Words')

    # vetically align the text of each topic
    plt.xticks(rotation=90) 
    
    # display the plot
    plt.show()    

drawNegPic(neg_text, 50)