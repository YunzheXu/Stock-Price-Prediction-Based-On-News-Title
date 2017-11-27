
# -*- coding: UTF-8 -*-  
from pymongo import MongoClient
from dateutil.parser import parse
from datetime import datetime, timedelta
import xlrd
import os
client = MongoClient('mongodb://localhost:27017/')
db = client['stock']
newsData = db['news']



path = "./"
# files= os.listdir(path)  
# for file in files:
# 	if file[-3:]=='xls':
# 		# data = pandas.read_csv(path+file)
# 		book = xlrd.open_workbook(path+file) 
# 		sh = book.sheet_by_index(0)
# 		print sh.nrows
# 		for row in range(1,sh.nrows):
# 			# print sh.cell_value(rowx=row, colx=0)
# 			# print sh.cell_value(rowx=row, colx=17)
# 			try:
# 				newsData.insert_one({'title':str(sh.cell_value(rowx=row, colx=0)),'date':parse(str(sh.cell_value(rowx=row, colx=17)))})
# 			except Exception, e:
# 				print e
			




# print newsData.find({'date':{'$lte': parse('2010-01-02'), '$gte': parse('2010-01-01')}}).count()
# titles = [a for a in newsData.find({'date':{'$lte': parse('2010 01, Jan')}})]
# print newsData.find({'date':{'$lte': parse('2010 01, Jan')}})
# print titles




# for a in newsData.find({'date':{'$lte': parse('2010-02-24').isoformat(), '$gte': parse('2010-01-24').isoformat()}}):
# 	print a

# for a in newsData.find({}):
# 	print a

# print newsData.delete_many({})







#########
a = {parse("2010-01-01"):"a",parse("2010-01-03"):"b"}

print parse("2010-01-01")+timedelta(days=3) not in a


