# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:39:05 2017

@author: Yunzhe
"""

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait # available since 2.4.0
from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
import time
import csv                                       
#Note : Because I used chrome as the browser, make sure you have installed chrome browser in your computer and had the chromedriver.exe in your folder before you run the file!
#If you don't have chromedriver, you can download from here: https://sites.google.com/a/chromium.org/chromedriver/downloads

# Create a new instance of the Chrome driver
driver = webdriver.Chrome()
driver.set_window_size(1920, 1080)
driver.maximize_window()

# go to the page
driver.get("http://search.ebscohost.com")
print ("We get the EBSCO page!")

# setting
ebsco_button = driver.find_element_by_xpath("//*[@id='tbProfiles']/tbody/tr[1]/td/table/tbody/tr/td[2]/a[1]")
ebsco_button.click()

select_all_button = driver.find_element_by_xpath("//*[@id='ctl00_ctl00_MainContentArea_MainContentArea_selectAll']")
select_all_button.click()

continue_button = driver.find_element_by_xpath("//*[@id='ctl00_ctl00_MainContentArea_MainContentArea_continue1']")
continue_button.click()

inputElement = driver.find_element_by_xpath("//*[@id='Searchbox1']")
inputElement.send_keys("google")

search_button = driver.find_element_by_xpath("//*[@id='SearchButton']")
search_button.click()

time.sleep(1);

news_button = driver.find_element_by_xpath("//*[@id='_doc_type_350NP']")
news_button.click()

time.sleep(1);

page_option_button = driver.find_element_by_xpath("//*[@id='lnkPageOptions']")
page_option_button.click()

detailed_button = driver.find_element_by_xpath("//*[@id='pageOptions']/li[1]/ul/li[4]/a") 
detailed_button.click()

page_option_button = driver.find_element_by_xpath("//*[@id='lnkPageOptions']")
page_option_button.click()

page_50_button = driver.find_element_by_xpath("//*[@id='pageOptions']/li[3]/ul/li[6]/a") 
page_50_button.click()

page_number = 1
number_per_page = 1
number = 1

# write csv file
 
with open("news titles from EBSCO.csv", "wb") as f:  
    # write to a csv file delimited 
    # by "\t" (you can set "," or other delimiters)                b
    writer=csv.writer(f, delimiter=',') 
    while page_number <= 20 :
        print ("Page number : ", page_number)
        
        for number_per_page in range(1, 51) :
            number = (page_number - 1) * 50 + number_per_page
            
            selector_title = "#Result_" + str(number)
            li_title = driver.find_elements_by_css_selector(selector_title)
            title = li_title[0].text.encode("utf-8","ignore")
    
            selector_detail = "#resultListControl > ul > li:nth-child(" + str(number_per_page) + ") > div > div"
            li_detail = driver.find_element_by_css_selector(selector_detail)
            detail = li_detail.text.encode("utf-8","ignore")
            
            
            row = [str(number), title, detail]
            writer.writerow(row)   
            
            print (page_number, number, title)
            
            number_per_page += 1;
            if number_per_page == 51 :
                number_per_page = 1
                break
        
        print ("Go to next page!")
        time.sleep(1)
        
        next_button = driver.find_element_by_xpath("//*[@id='ctl00_ctl00_MainContentArea_MainContentArea_bottomMultiPage_lnkNext']")
        next_button.click()    
        page_number += 1
    
f.close()

print ("Done!")
print ("The driver will be closed in 5s")

print ("5")
time.sleep(1)
print ("4")
time.sleep(1)
print ("3")
time.sleep(1)
print ("2")
time.sleep(1)
print ("1")
time.sleep(1)
print ("Closed!")
driver.quit()





