from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from time import sleep
import pdb

def startup():
    # Using Chrome to access web
    #driver = webdriver.Chrome()
    
    chromeOptions = Options()
    # Open the browser in full screen
    chromeOptions.add_argument("--window-size=1920,1080")
    # Don't show automation mode
    chromeOptions.add_experimental_option("excludeSwitches", ["enable-automation"])
    chromeOptions.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(chrome_options=chromeOptions)

    # Get the website
    driver.get('http://www.i-ppi.jp/IPPI/SearchServices/Web/Koji/Kokoku/Search.aspx')

    return driver

def selectByvalue(driver, name, value):
	select = Select(driver.find_element_by_name(name))
	select.select_by_value(value)
	sleep(1)

def getPDFurl(driver):
	aa = driver.find_element_by_class_name('searchgrid_bunsho')
	
	return aa.find_element_by_xpath('tbody/tr/td/a').get_attribute('href')

def postBid(driver, pdf):

	for i in range(1, 21):
		links = driver.find_elements_by_xpath('html/body/form/div/div/table[@class="searchgrid"]/tbody/tr')
		links[i].find_elements_by_xpath('td/a')[0].click()
		sleep(0.5)
		pdf += [getPDFurl(driver)]
		driver.back()

def url_to_txt(pdf):
	with open('pdfurl2.txt', 'w') as file:
		for url in pdf:
			file.write("%s\n" % url) 

driver = startup()

# select the start year
selectByvalue(driver, 'drpKokokuYearFrom', '2016')
selectByvalue(driver, 'drpKokokuMonthFrom', '01')
selectByvalue(driver, 'drpKokokuDayFrom', '01')
selectByvalue(driver, 'drpCount', '20')

selectByvalue(driver, 'drpKokokuMonthTo', '03')
selectByvalue(driver, 'drpKokokuDayTo', '27')

# start_index
search = driver.find_element_by_id('btnSearch')
search.click()
sleep(1)



pdf = []
# Get links with hyperlink
# links = driver.find_elements_by_xpath('html/body/form/div/div/table[@class="searchgrid"]/tbody/tr')
# links[1].find_elements_by_xpath('td/a')[0].click()
while True:
	postBid(driver, pdf)
	url_to_txt(pdf)
	next_page = driver.find_element_by_id('btnNext1')
	try:
		next_page.click()
	except ElementNotInteractableException:
		print('end up!')
		break


pdb.set_trace()
















