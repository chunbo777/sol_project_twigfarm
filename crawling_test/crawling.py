### https://brunch.co.kr/ - Crawling

### Library import
import re
import json
import requests
import urllib
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random
import pickle
import nltk
import kss
import pandas as pd
import numpy as np


### profileId crawler
'''
    *content에 접근하기 위한 userID 추출
    *publishTime으로 여러 게시물 로드
    *한 페이지당 20개의 게시물(20개 이하의 아이디 추출 및 중복 아이디 제거)
'''
### 브런치 카테고리 "지구한바퀴 세계여행(/keyword/38)"
# url & headers
profile_url = 'https://api.brunch.co.kr/v1/top/keyword/group/52?publishTime={}&pickContentId='
profile_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'}

#url_list = ['33','34','52','67']
'''
2 : 글쓰기 코치, 9: 직장인 현실 조언
13 : 뮤직 인사이드
32 : 영화리뷰, 33 : IT/트랜드, 34 : 그림/웹툰, 39 : 사진/촬영
40 : 책
52 : 인문학/철학
67 : 시사/이슈
'''

# 게시물 로드에 이용할 publishTime 변수
int_time = 1696768787365
str_time = "1696768787365"

# profileId를 담을 변수
profile_list = []
# 전처리가 완료된 데이터를 담을 변수
cleaned_result = []
for i in tqdm(range(0, 1000)):#1000번의 페이지 바꾸기
    for j in range(0, 19): #페이지당 20명의 유저 아이디 추출(0~19)
        profile_url = 'https://api.brunch.co.kr/v1/top/keyword/group/52?publishTime={}&pickContentId='.format(str_time)
        variables = {"publishTime":str_time
        }
        profile_params = {"variables":json.dumps(variables)
        }

        respones = requests.get(profile_url, params=profile_params, headers=profile_headers).json()
        #js = json.loads(rs.text)
        
        #중복값 제거를 위한 리스트
        delete = []
        for t in range(0, 1):
            profileId = respones['data']['articleList'][t]['article']['profileId']
            delete.append(profileId)
            print(str_time)
            for overlap in delete:
                if overlap not in profile_list:
                    profile_list.append(overlap)
                    print("아이디를 추출했습니다. →", profile_list)
                    #크롤러 위장 코드
                    time.sleep(random.uniform(2,6))  

                    #앞에서 받아온 유저명을 가지고 게시물 접근
                    for _q in profile_list:
                        q = urllib.parse.quote(_q)
                        for g in range(1, 400): #해당 유저명에서 1부터 400까지의 게시물을 확인 및 크롤링
                            variables = {"q":q
                                        ,"g":g}
                            params = {"variables":variables}

                            url = "https://brunch.co.kr/@{}/{}".format(q,g)
                            headers={
                                'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'
                            }
                            
                            response = requests.get(url, params=params, headers=headers)

                            #status code 값이 200(정상)이면 해당 게시물을 크롤링
                            if response.status_code == 200:
                                print(f"{q}의 {g}번째 게시물에 접근", "=", "☆★☆★",response.status_code,"☆★☆★")
                                html = response.text
                                bs = BeautifulSoup(html, 'html.parser')
                                print("profileId : ",q,",", "현재 페이지 :",g)
                                # 크롤러를 위장하는 코드
                                time.sleep(random.uniform(2,6)) 

                                h4 = bs.select('h4')
                                print("요소 태그에 접근합니다.")    
                                
                                #해당 사이트는 데이터가 h4태그 또는 p태그에 존재하므로
                                #데이터가 들어 있는 태그에 접근해 데이터를 추출
                                result_h4 = []
                                if len(h4) == 0: #h4 태그에 데이터가 없을 경우 p태그 추출 코드로 진행
                                    print("<p> 태그에 접근했습니다.")

                                    #p태그 안의 데이터 추출
                                    p = bs.find_all('p')

                                    # 크롤러를 위장하는 코드
                                    time.sleep(random.uniform(2,6)) 
                                    

                                    ### 데이터 전처리
                                    try:
                                        result = []
                                        for s in p:
                                            s = s.get_text().strip()
                                            if len(s) != 0:
                                                result.append(s)
                                    
                                        cleaned_p = []
                                        for sent in result:
                                            for sent in kss.split_sentences(sent):
                                                cleaned_p.append(sent)
                                    
                                        for p_len in range(len(cleaned_p)):
                                            if len(cleaned_p[p_len]) > 20 and len(cleaned_p[p_len]) < 51:
                                                result.append(cleaned_p[p_len])
                                                print(f"Insert {p_len}번")
                                                #데이터 저장
                                                with open('/home/tf-dev-01/workspace_sol/style-transfer/crawling/keyword_52.csv', 'a') as f:
                                                    f.writelines(cleaned_p[p_len])
                                                    f.write('\n')
                                                    f.close()
                                                print(f"Complete {p_len}번", cleaned_p[p_len])

                                    except Exception as e:
                                        print(e, "발생!")
                                        result = []
                                        for s in p:
                                            s = s.get_text().strip()
                                            if len(s) != 0:
                                                result.append(s)

                                        result = str(result).strip().replace('. ', '.\n')
                                        with open('/home/tf-dev-01/workspace_sol/style-transfer/crawling/error_52.csv', 'a') as f:
                                            f.writelines(result)
                                            f.write('\n')
                                            f.close()
                                            print(e, "발생으로 error.csv 저장완료")

                                #h4 태그에 데이터가 존재할 경우 아래 코드 실행
                                else:
                                    print("<h4> 태그에 접근했습니다.")
                                    #h4 태그의 데이터 추출
                                    h4 = bs.select('h4')
                                    # 크롤러 위장
                                    time.sleep(random.uniform(2,6)) 
                                    
                                    result = []
                                    for s in h4:
                                        s = s.get_text().strip()
                                        if len(s) != 0:
                                            result.append(s)
                                    
                                    cleaned_h4 = []
                                    for sent in result:
                                        for sent in kss.split_sentences(sent):
                                            cleaned_h4.append(sent)
                                    
                                    for h4_len in range(len(cleaned_h4)):
                                        if len(cleaned_h4[h4_len]) > 20 and len(cleaned_h4[h4_len]) < 51:
                                            result.append(cleaned_h4[h4_len])
                                            print(f"Insert {h4_len}번")
                                            with open('/home/tf-dev-01/workspace_sol/style-transfer/crawling/keyword_52.csv', 'a') as f:
                                                f.writelines(cleaned_h4[h4_len])
                                                f.write('\n')
                                                f.close()
                                            print(f"Complete {h4_len}번", cleaned_h4[h4_len])
                            # status code가 400(에러)일 경우 해당 아이디와 게시물 번호를 출력하고 continue
                            else : 
                                print(f"{q}의 {g}번째 게시물에 접근", "=", response.status_code)
                                continue
                                
    #int_time -= 10000000000
    int_time -= 9999999999
    int_time -= 7468468768
    str_time = str(int_time)
    print("유저 목록:",profile_list)
    #print("{}번째 아이디 : {}의 {}번째 게시물을 출력을 완료했습니다.").format(len(profile_list),profile_list[i],g)
    print(len(profile_list),"번째 아이디 :", profile_list[i],"의", g,"번째 게시물을 출력을 완료했습니다.")
    print(i)


