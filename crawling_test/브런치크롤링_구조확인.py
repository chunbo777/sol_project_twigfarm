import requests
import json
from bs4 import BeautifulSoup

url = 'https://brunch.co.kr/keyword/%EC%A7%80%EA%B5%AC%ED%95%9C%EB%B0%94%ED%80%B4_%EC%84%B8%EA%B3%84%EC%97%AC%ED%96%89?q=g'

headers = {'users-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'}

r = requests.get(url, headers=headers)

soup = BeautifulSoup(r.content, 'lxml')
div = soup.find('div', id='wrapArticle')
div

import requests
import json
import time

publishTime = 1631457213000
pickContentId = '' 

for i in range(10): # 0~9

    params = {
        'publishTime': publishTime
    }

    response = requests.get("https://api.brunch.co.kr/v1/top/keyword/group/38?publishTime=1631457213000&pickContentId=", params=params)
    data = json.loads(response.text)
    
    test = data['data']
    test['articleList']

    for d in test['articleList']:
    print(d['contentSummary'])

    publishTime = data['publishTime']

    time.sleep(1)

test = data['data']
test

test2 = test['articleList']
test3 = test2.get('contentSummary')
test3

for d in test['articleList']:
    print(d['contentSummary'])

    publishTime = data['publishTime'] #0


    time.sleep(1)

    

#print(data['moreList'])

for d in data['data']:
        print(d['keyword'])

    publishTime = data['publishTime'] #0


    time.sleep(1)

    

#print(data['moreList'])

import requests
import json
import time

magazineOffset = 0
contestOffset = 0
exhibitOffset = 0
galleryOffset = 0

for i in range(10): # 0~9

    params = {
        'magazineOffset': magazineOffset
        ,'contestOffset': contestOffset
        ,'exhibitOffset': exhibitOffset
        ,'galleryOffset': galleryOffset
    }

    response = requests.get("https://www.jungle.co.kr/recent.json", params=params)
    data = json.loads(response.text)

for d in data['moreList']:
        print(d['title'])
        print(d['targetCode'])

    magazineOffset = data['magazineOffset'] #0
    contestOffset = data['contestOffset'] #6
    exhibitOffset = data['exhibitOffset'] #0
    galleryOffset = data['galleryOffset'] #0

    time.sleep(1)

    

#print(data['moreList'])

