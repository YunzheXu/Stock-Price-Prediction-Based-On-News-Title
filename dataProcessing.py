from dateutil.parser import parse
import pandas as pd
import numpy as np
import time
import nltk
import re
import matplotlib.pyplot as plt
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['stock']
newsData = db['news']


def priceList():
	df = pd.read_csv('./goog.csv')
	return df

def getUpDownList():
	priceList = {}
	df = pd.read_csv('./goog.csv')
	for a in range(0,len(df['Date'])-1):
		if df['Close'][a] > df['Close'][a+1]:
			priceList[parse(df['Date'][a])] = 1
		else:
			priceList[parse(df['Date'][a])] = 0
	# print('price list:%d'% len(priceList))
	return priceList

def getPriceListBeforeDate(bDate):
	price = []
	x = priceList()
	for i in range(0,len(priceList()['Date'])):
		if parse(x['Date'][i])<parse(bDate):
			price.append({'price':x['Close'][i],'Date':x['Date'][i]})
	print('price list:%d'% len(price))
	return price

# price = [x for x in priceList() if parse(x['Date'])<parse('1-Feb-10')]
# print price


def getTitles(bDate):
	df = pd.DataFrame([a for a in newsData.find({'date':{'$lte': parse(bDate)}})])
	# pattern = re.compile(r"\w+ \d+, \d+")
	titles = []
	days = []
	# for d in df['pubdate']:
	# 	if pattern.match(d) != None:
	# 		days.append(parse(d))
	daySet = set([d['Date'] for d  in getPriceListBeforeDate(bDate) ])
	# print np.array(daySet)
	for d in daySet:
		tempTitle = []
		for row in range(0,len(df)):
			# if pattern.match(df['date'][row]) != None:
			if df['date'][row] == parse(d):
				tempTitle.append(df['title'][row])
		# tokens=nltk.word_tokenize("".join(titles).lower())
		titles.append({'Date':parse(d),'Title':"".join(tempTitle)})
	return titles
	




def plotPrice(bDate):
	sentimentData = getSentiment(bDate)
	priceData = getPriceListBeforeDate(bDate)
	X = []
	count = 0
	price = []
	titles = []
	dateArray =[]
	for i in range(0,len(priceData)):
		dateArray.append(parse(priceData[i]['Date']))
		X.append(i)
		price.append(priceData[i]['price'])
	
	# maxPrice = np.max(price)
	# for i in range(0,len(priceData)):
	# 	titles.append(sentimentData[i])
	fig = plt.figure()

	ax1 = fig.add_subplot(111)
	ax1.plot(X, price)
	ax1.set_ylabel('price')
	ax1.set_title("sentiment analysis and stock price")
	print len(X)
	print len(price)
	print len(sentimentData)
	ax2 = ax1.twinx()  # this is the important function
	ax2.plot(X, sentimentData, 'r')
	# ax2.set_xlim([0, np.e])
	ax2.set_ylabel('sentiment analysis data')
	ax2.set_xlabel('time')

	temp = pd.DataFrame({"Date":dateArray,"Sentiment":sentimentData,"price":price})
	temp.to_csv('Sentiment.csv')

	plt.show()# show the plot on the screen

from nltk.sentiment.vader import SentimentIntensityAnalyzer



def getSentiment(bDate):
	compon = []
	titles = getTitles(bDate)
	sid = SentimentIntensityAnalyzer()
	for sentence in titles:
		ss = sid.polarity_scores(sentence["Title"])
		compon.append(ss['compound'])
	return compon

# plotPrice('2011 01, Jan')


import nltk, re, string
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
# numpy is the package for matrix cacluation
import numpy as np  

# Step 1. get tokens of each document as list
def get_doc_tokens(doc):
    stop_words = stopwords.words('english')
    tokens=[token.strip() \
            for token in nltk.word_tokenize(doc.lower()) \
            if token.strip() not in stop_words and\
               token.strip() not in string.punctuation]
    
    # you can add bigrams, collocations, or lemmatization here
    
    return tokens

def tfidf(docs):
    # step 2. process all documents to get list of token list
    docs_tokens=[get_doc_tokens(doc["Title"]) for doc in docs]
    voc=list(set([token for tokens in docs_tokens \
              for token in tokens]))

    # step 3. get document-term matrix
    dtm=np.zeros((len(docs), len(voc)))

    for row_index,tokens in enumerate(docs_tokens):
        for token in tokens:
            col_index=voc.index(token)
            dtm[row_index, col_index]+=1
            
    # step 4. get normalized term frequency (tf) matrix        
    doc_len=dtm.sum(axis=1, keepdims=True)
    tf=np.divide(dtm, doc_len)
    
    # step 5. get idf
    doc_freq=np.copy(dtm)
    doc_freq[np.where(doc_freq>0)]=1

    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(doc_freq, axis=0)+1))+1

    
    # step 6. get tf-idf
    smoothed_tf_idf=normalize(tf*smoothed_idf)
    
    return smoothed_tf_idf



titles = [a for a in getTitles('2011 01, Jan')]
dates = [a["Date"].strftime("%Y-%m-%d") for a in titles]
print dates
df = pd.DataFrame({"title":[",".join(str(a)) for a in tfidf(titles)],"date":[a["Date"].strftime("%Y-%m-%d") for a in titles]})
writer = pd.ExcelWriter('output.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()




