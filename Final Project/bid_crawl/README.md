# bid webcrawling

1. please download webdriver [here](https://chromedriver.chromium.org/downloads).

	This will be used to start the Google Chrome while web-crawling.

2. Make sure you are downloading the webdrive version is corresponding to your google chrome version.

3. For install selenium:

	`pip install selenium`

4. Start webcrawling:

	`python3 bid_crawl.py`

5. Everything should be OK. And the urls to target pdf will be stored in the pdfurl.txt

6. Download all the pdf from url in pdfurl.txt:

	`wget -i pdfurl -P PDF/`

	All the pdf file should be in PDF folder.