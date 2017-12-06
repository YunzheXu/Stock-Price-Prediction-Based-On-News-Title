from dateutil.parser import parse
import time
import nltk
import re
import matplotlib.pyplot as plt
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['stock']
newsData = db['news']
import pandas as pd
import numpy as np
import nltk,string
from gensim import corpora
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, \
Dropout, Activation, Input, Flatten, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


def cnn_model(FILTER_SIZES, \
              # filter sizes as a list
              MAX_NB_WORDS, \
              # total number of words
              MAX_DOC_LEN, \
              # max words in a doc
              NUM_OUTPUT_UNITS=1, \
              # number of output units
              EMBEDDING_DIM=200, \
              # word vector dimension
              NUM_FILTERS=64, \
              # number of filters for all size
              DROP_OUT=0.5, \
              # dropout rate
              PRETRAINED_WORD_VECTOR=None,\
              # Whether to use pretrained word vectors
              LAM=0.01):            
              # regularization coefficient
    
    main_input = Input(shape=(MAX_DOC_LEN,), \
                       dtype='int32', name='main_input')
    
    if PRETRAINED_WORD_VECTOR is not None:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        weights=[PRETRAINED_WORD_VECTOR],\
                        trainable=False,\
                        name='embedding')(main_input)
    else:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        name='embedding')(main_input)
    # add convolution-pooling-flat block
    conv_blocks = []
    for f in FILTER_SIZES:
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, \
                      activation='relu', name='conv_'+str(f))(embed_1)
        conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
        conv = Flatten(name='flat_'+str(f))(conv)
        conv_blocks.append(conv)

    z=Concatenate(name='concate')(conv_blocks)
    drop=Dropout(rate=DROP_OUT, name='dropout')(z)

    dense = Dense(192, activation='relu',\
                    kernel_regularizer=l2(LAM),name='dense')(drop)
    preds = Dense(NUM_OUTPUT_UNITS, activation='sigmoid', name='output')(dense)
    model = Model(inputs=main_input, outputs=preds)
    
    model.compile(loss="binary_crossentropy", \
              optimizer="adam", metrics=["accuracy"]) 
    
    return model

def getMAX_NB_WORDS():
		# total number of words
	total_nb_words=len(tokenizer.word_counts)
	print(total_nb_words)

	# put word and its counts into a data frame
	word_counts=pd.DataFrame(\
	            tokenizer.word_counts.items(), \
	            columns=['word','count'])
	word_counts.head(3)

	# get histogram of word counts
	# after reset index, "index" column 
	# is the word frequency
	# "count" column gives how many words appear at 
	# a specific frequency
	df=word_counts['count'].value_counts().reset_index()
	df.head(3)

	# convert absolute counts to precentage
	df['percent']=df['count']/len(tokenizer.word_counts)
	# get cumulative percentage
	df['cumsum']=df['percent'].cumsum()
	df.head(5)

	# plot the chart
	# chart shows >90% words appear in less than 50 times
	# if you like to include only words occur more than 50 times
	# then MAX_NB_WORDS = 10% * total_nb_words
	plt.bar(df["index"].iloc[0:50], df["percent"].iloc[0:50])
	plt.plot(df["index"].iloc[0:50], df['cumsum'].iloc[0:50], c='green')

	plt.xlabel('Word Frequency')
	plt.ylabel('Percentage')
	plt.show()

def getMAX_DOC_LEN():
		# include complete sentences as many as possible

	# create a series based on the length of all sentences
	sen_len=pd.Series([len(item) for item in sequences])

	# create histogram of sentence length
	# the "index" is the sentence length
	# "counts" is the count of sentences at a length
	df=sen_len.value_counts().reset_index().sort_values(by='index')
	df.columns=['index','counts']
	df.head(3)

	# sort by sentence length
	# get percentage and cumulative percentage

	df=df.sort_values(by='index')
	df['percent']=df['counts']/len(sen_len)
	df['cumsum']=df['percent'].cumsum()
	df.head(3)

	# From the plot, 90% sentences have length<500
	# so it makes sense to set MAX_DOC_LEN=4~500 
	plt.plot(df["index"], df['cumsum'], c='green')

	plt.xlabel('Sentence Length')
	plt.ylabel('Percentage')
	plt.show()

def createCNN():
	from sklearn.model_selection import train_test_split

	# set the number of output units
	# as the number of classes
	NUM_OUTPUT_UNITS=len(mlb.classes_)

	EMBEDDING_DIM=100
	FILTER_SIZES=[2,3,4]
	BEST_MODEL_FILEPATH='best_model'
	BTACH_SIZE = 64
	NUM_EPOCHES = 20

	# split dataset into train (70%) and test sets (30%)
	X_train, X_test, Y_train, Y_test = train_test_split(\
	                padded_sequences, Y, test_size=0.3, random_state=0)


	model=cnn_model(FILTER_SIZES, MAX_NB_WORDS, \
	                MAX_DOC_LEN, NUM_OUTPUT_UNITS)

	earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
	checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_acc', \
	                             verbose=2, save_best_only=True, mode='max')
	    
	training=model.fit(X_train, Y_train, \
	          batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
	          callbacks=[earlyStopping, checkpoint],\
	          validation_data=[X_test, Y_test], verbose=2)


	# load the best model
	model.load_weights("best_model")

	pred=model.predict(X_test)
	pred[0:5]

	# Exercise 5.7.6: Generate performance report
	from sklearn.metrics import classification_report

	# create a copy of the predicated probabilities
	Y_pred=np.copy(pred)
	# if prob>0.5, set it to 1 else 0
	Y_pred=np.where(Y_pred>0.5,1,0)

	Y_pred[0:10]
	Y_test[0:10]

	print(classification_report(Y_test, Y_pred, \
	                            target_names=mlb.classes_))

	

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

if __name__ == '__main__':

	titles = [a for a in getTitles('2011 01, Jan')]
	dates = [a["Date"] for a in titles]
	priceList = getUpDownList()
	labels = []
	text = []
	for a in titles:
		if a["Date"] in priceList:
			text.append(a["Title"])
			labels.append(priceList[a["Date"]])

	# data=pd.read_csv("amazon_review_large.csv", header=None)
	labels = [str(a) for a in labels]
	# text = data[1]
	# text = list(text)

	mlb = MultiLabelBinarizer()
	Y=mlb.fit_transform(labels)


	# MAX_NB_WORDS :47008
	MAX_NB_WORDS=26825
	# documents are quite long in the dataset
	MAX_DOC_LEN=3400

	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(text)
	voc=tokenizer.word_index
	# convert each document to a list of word index as a sequence
	sequences = tokenizer.texts_to_sequences(text)
	# get the mapping between words to word index

	# pad all sequences into the same length (the longest)
	padded_sequences = pad_sequences(sequences, \
	                                 maxlen=MAX_DOC_LEN, \
	                                 padding='post', truncating='post')
	createCNN()
	# getMAX_NB_WORDS()
	# getMAX_DOC_LEN()


