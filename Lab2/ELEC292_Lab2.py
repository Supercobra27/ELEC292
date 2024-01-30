# Ryan
import requests
from bs4 import BeautifulSoup

http_text = requests.get("https://weather.com/en-CA/weather/tenday/l/584018bec07ce9573837c14fa59da031fa6fcdeb1c3c9e3b2b27cb79ce254b5a").text # send GET request to website

soup = BeautifulSoup(http_text, 'lxml')
weather_data = soup.find_all('div', class_="DetailsSummary--DetailsSummary--1DqhO DetailsSummary--fadeOnOpen--KnNyF") # find all scrapable tags

for day in weather_data: #iterate over
    date = day.find('h3', class_="DetailsSummary--daypartName--kbngc").text # find the date of each item
    span_tags = day.find('div', class_="DetailsSummary--temperature--1kVVp").find_all('span') # gets the temperature span tags
    max_temp = span_tags[0].text.replace('°','') #get max and get rid of degree
    min_temp = span_tags[3].text.replace('°','') #get min and get rid of degree
    weather_condition = day.find('div', class_="DetailsSummary--condition--2JmHb").span.text #get condition
    precip_chance = day.find('div', class_="DetailsSummary--precip--1a98O").span.text #get rain chaine
    wind_section = day.find('div', class_="DetailsSummary--wind--1tv7t DetailsSummary--extendedData--307Ax").span.text.split() #get wind data
    final_data = (date, max_temp, min_temp, weather_condition, precip_chance, wind_section[0], wind_section[1]) #consolidate
    with open('ELEC292_Lab2.md', 'a') as f: #Open file
        print(final_data, file=f) #print to file
