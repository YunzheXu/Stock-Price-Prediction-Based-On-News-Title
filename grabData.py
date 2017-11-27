from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.select import Select
import time

def preSelect(companyName):
	driver = webdriver.Chrome('./chromedriver')
	driver.get('https://search.proquest.com/abicomplete/advanced?accountid=14052')
	driver.find_element_by_name('queryTermField').send_keys(companyName)
	driver.find_element_by_name('itemsPerPage').click()
	s1 = Select(driver.find_element_by_id('itemsPerPage'))
	s1.select_by_visible_text("100") 
	driver.find_element_by_name('queryTermField').send_keys(Keys.RETURN)

	time.sleep(2)

	
	# type news
	driver.find_element_by_id('objectype-header').click()
	time.sleep(1)
	driver.find_element_by_id('filter_11').click()
	time.sleep(1)
	# english only
	driver.find_element_by_id('language-header').click()
	time.sleep(1)
	driver.find_element_by_id('filter_11').click()
	driver.find_element_by_id('upDateDateRangeLink').click()
	return driver


def getdata(startDate,endDate,driver,page):
	driver.find_element_by_id('startingDate').clear()
	driver.find_element_by_id('endingDate').clear()
	driver.find_element_by_id('startingDate').send_keys(startDate)
	driver.find_element_by_id('endingDate').send_keys(endDate)
	time.sleep(1)
	driver.find_element_by_id('dateRangeSubmit').click()
	time.sleep(1)

	import string
	while page<=int(filter(lambda c: c in string.digits + '.', driver.find_element_by_id('pqResultsCount').text))/100+1:
		driver.find_element_by_id('mlcbAll').click()
		if page > 9 and page%10 == 0:
			print "get Excel !!!"
			driver.find_element_by_id('tsMore').click()
			driver.find_element_by_id('saveExportLink_6').click()
			time.sleep(10)
			driver.find_element_by_xpath('//*[@id="modal-footer-ux"]/div[1]/div/a').click()
			time.sleep(30)
			driver.find_element_by_id('selecteditemsclear_link').click()
			try:
				time.sleep(5)
				driver.find_element_by_id('button_3').click()
			except Exception, e:
				return page
			
		page += 1
		print page
		url = list(driver.current_url.strip('#'))
		print "".join(url[:67])+str(page)+"".join(url[-16:])
		driver.get("".join(url[:67])+str(page)+"".join(url[-16:]))
		
	driver.find_element_by_id('tsMore').click()
	driver.find_element_by_id('saveExportLink_6').click()
	time.sleep(5)
	driver.find_element_by_xpath('//*[@id="modal-footer-ux"]/div[1]/div/a').click()
	time.sleep(25)
	return 0


def getAnnualData(companyName):
	driver = preSelect(companyName)
	for i in range(2,12):
		startDate = ''
		endDate = ''
		page = 1
		if i< 10:
			startDate = '2011-0'+str(i)+'-01'
			if i == 9:
				endDate = '2011-10-01'
			else:	
				endDate = '2011-0'+str(i+1)+'-01'
		else:
			startDate = '2011-'+str(i)+'-01'
			endDate = '2011-'+str(i+1)+'-01'
		print('%s  %s' % (startDate,endDate))
		page = getdata(startDate,endDate,driver,page)
		if page == 0:
			driver = preSelect(companyName)
		else:
			driver.get(driver.current_url.strip('#'))
			getdata(startDate,endDate,driver,page)


getAnnualData('google')