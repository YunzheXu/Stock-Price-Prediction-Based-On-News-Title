# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:40:12 2017

@author: Yunzhe
"""

import csv
import re
from nltk.corpus import stopwords
from string import punctuation

processed_data = []
number = 1
stop_words = stopwords.words('english')

#read raw data
with open("news titles from EBSCO.csv", "rb") as rf:  
    # write to a csv file delimited 
    # by "\t" (you can set "," or other delimiters)                b
    reader=csv.reader(rf, delimiter=',')
    for row in reader:
        #get publication date by regex and remove news without date
        match = re.findall(r'(\d+/\d+/\d+)', row[2])
        if (match != []):
            date = match[0]
        else:
            continue
        
        #lower the words and strip punctuaction
        mystr = row[1].lower()
        words = re.sub("[^\w]", " ",  mystr).split()
        #remove stop words in title
        filtered_words = [word for word in words if word not in stop_words]
        title = ""
        for word in filtered_words:
            title = title + word + " "
        title.strip()

        processed_data.append([str(number), title, str(date)])
        number += 1
    rf.close()

#save the data
with open("cleaned and preprocessed data from EBSCO.csv", "wb") as wf:
    writer = csv.writer(wf, delimiter=',')
    for row in processed_data:
        writer.writerow(row)
    wf.close()

print ("done!")
