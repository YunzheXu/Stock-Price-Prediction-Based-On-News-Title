from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from dateutil.parser import parse
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import math
import os
import pandas as pd
import numpy as np

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

# def get_doc_tokens(doc):
#     stop_words = stopwords.words('english')
#     tokens=[token.strip() \
#             for token in nltk.word_tokenize(doc.lower()) \
#             if token.strip() not in stop_words and\
#                token.strip() not in string.punctuation]
    
#     # you can add bigrams, collocations, or lemmatization here
    
#     return tokens

# def tfidf(docs):
#     # step 2. process all documents to get list of token list
#     docs_tokens=[get_doc_tokens(doc) for doc in docs]
#     voc=list(set([token for tokens in docs_tokens \
#               for token in tokens]))

#     # step 3. get document-term matrix
#     dtm=np.zeros((len(docs), len(voc)))

#     for row_index,tokens in enumerate(docs_tokens):
#         for token in tokens:
#             col_index=voc.index(token)
#             dtm[row_index, col_index]+=1
            
#     # step 4. get normalized term frequency (tf) matrix        
#     doc_len=dtm.sum(axis=1, keepdims=True)
#     tf=np.divide(dtm, doc_len)
    
#     # step 5. get idf
#     doc_freq=np.copy(dtm)
#     doc_freq[np.where(doc_freq>0)]=1

#     smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(doc_freq, axis=0)+1))+1

    
#     # step 6. get tf-idf
#     smoothed_tf_idf=normalize(tf*smoothed_idf)
    
#     return smoothed_tf_idf
def getSentimentList():
	priceList = getUpDownList()
	df = pd.read_csv('Sentiment10.csv')
	sentiment = [] 
	price = []
	for row in range(30,len(df["Sentiment"])-1):
		try:
			counter = 0
			tempSentimentList = []
			while len(tempSentimentList)<21:
				if parse(df["Date"][row])-timedelta(days=counter) in priceList:
					# sentiment.append(df["Sentiment"][row])
					tempSentimentList.append(df["Sentiment"][row-counter]*math.exp(counter))
				counter += 1
				if counter>10000:
					print counter
			sentiment.append(tempSentimentList)
			price.append(priceList[parse(df["Date"][row+1])])
		except Exception, e:
			print e

	return sentiment,price



sentiment,price = getSentimentList()



# def getTfidfList():
# 	df = pd.read_csv('tfidf.csv')
# 	temp = []
# 	for row in df.iterrows():
# 		index, data = row
# 		temp.append(data.tolist())

# 	return temp


# print getTfidfList()

sentiment = np.array(sentiment,dtype=float)
price = np.array(price,dtype=int)


price = price.reshape(-1, )
sentiment = sentiment.reshape(-1, 21)

print price.shape
print sentiment.shape

train = int(len(price)*0.75)
predict = len(price) - train

# temp = train

# train = predict

# predict = temp


# clf = svm.SVC() #58.7301587302 		
# clf = GaussianNB() #55.5555555556
clf = KNeighborsClassifier(n_neighbors=150) #61.9047619048
# clf = DecisionTreeClassifier() #84.126984127	
# clf = LogisticRegression() #55.5555555556

# print price
# clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, sentiment, price, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# SVM : 0.51 (+/- 0.01)
# KNN : 0.55 (+/- 0.10)
# DecisionTree: 0.57 (+/- 0.10)
# LogisticRegression: 0.50 (+/- 0.15)


# clf.fit(sentiment[:train], price[:train])
# r = price[:predict]
# p = clf.predict(sentiment[:predict])
# count = 0.0
# for a in range(0,predict):
# 	if(r[a]==p[a]):
# 		count+=1
# total = []
# total.append(count/predict)
# print(sum(total)/float(len(total)) * 100)