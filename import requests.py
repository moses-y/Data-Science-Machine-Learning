import requests
from bs4 import BeautifulSoup

# The webpage URL from which you want to scrape the video
page_url = 'https://learn.business-science.io/webinar-replay-python?he=mosesyebei@gmail.com&el=email1'

# Send a GET request to fetch the page content
response = requests.get(page_url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.text, 'html.parser')
