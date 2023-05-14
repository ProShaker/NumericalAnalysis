# 네이버 뉴스 기사 크롤링  + 엑셀 파일로 정리하기 실습
# https://www.youtube.com/watch?v=lBviJv_njJk&list=PLDAF4b3hGb-HI3Emxb1MU6AvdomysgoQa&index=108
# https://youtu.be/66m9FjRo4cY

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from openpyxl import Workbook
import time

xlsx = Workbook()

# 1. 검색할 키워드 입력
print(1)
keyWord = input('검색 키워드 입력 : ')
time.sleep(1)

# 시트 생성
xlsx.create_sheet(keyWord)
sheet = xlsx[keyWord]
sheet.append(['제목', 'URL'])

# 2. 크롬드라이버로 원하는 url 접속
print(2)
url ='https://www.naver.com'
driver = webdriver.Chrome()
driver.get(url)
time.sleep(2)

# 3. naver 검색 창에 키워드 입력 후 enter
print(3)
search = driver.find_element(By.ID, 'query')
search.send_keys(keyWord)
search.send_keys(Keys.RETURN)
time.sleep(2)

# 4. 뉴스 탭 클릭
print(4)
driver.find_element(By.XPATH, '//*[@id="lnb"]/div[1]/div/ul/li[10]/a').click()
time.sleep(2)

# 5. 검색 결과 페이지에서 원하는 뉴스 기사 내용 수집

all_news = driver.find_elements(By.CLASS_NAME, 'news_tit')

for news in all_news:
    print(news.text)
    print(news.get_attribute('href'))
    title = news.text
    url = news.get_attribute('href')
    sheet.append([title, url])

driver.quit()

del xlsx['Sheet']
filename = keyWord + " 크롤링 데이터.xlsx"
xlsx.save(filename)
xlsx.close()
