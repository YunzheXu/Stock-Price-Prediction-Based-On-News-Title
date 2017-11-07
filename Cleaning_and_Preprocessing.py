# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:42:01 2017

@author: Yunzhe
"""

import csv
import re
import nltk
from nltk.corpus import stopwords
import string

processed_data = []
number = 1
stop_words = stopwords.words('english')

#read raw data
with open("news titles from EBSCO.csv", "rb") as rf:  
    reader=csv.reader(rf, delimiter=',')
    for row in reader:
        #get publication date by regex and remove news without date
        match = re.findall(r'(\d+/\d+/\d+)', row[2])
        if (match != []):
            date = match[0]
        else:
            continue
                
        #lower the words and do tokenization
        text = row[1].lower()

        pattern = r'[a-z]+[a-z\-\.]*'                        
        tokens = nltk.regexp_tokenize(text, pattern)

        #remove stop words 
        filtered_tokens = [token for token in tokens if token not in stop_words]
        
        title = ""
        for token in filtered_tokens:
            title = title + token + " " 
            
        title = title.strip()
        title = title.strip(string.punctuation)

        processed_data.append([str(number), str(title), str(date)])
        number += 1
    rf.close()

#save the data
with open("cleaned and preprocessed data from EBSCO.csv", "wb") as wf:
    writer = csv.writer(wf, delimiter=',')
    for row in processed_data:
        writer.writerow(row)
    wf.close()

print ("The raw data is cleaned and preprocessed!")
