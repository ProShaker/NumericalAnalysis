from bs4 import BeautifulSoup
from selenium import webdriver
import time

base_url = "https://search.naver.com/search.naver?where=view&sm=tab_jum&query="

keyword = input("검색어를 입력하세요 : ")

search_url = base_url + keyword

driver = webdriver.Chrome()

driver.get(search_url)

time.sleep(3)

for i in range(5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

html = driver.page_source

soup = BeautifulSoup(html, "html.parser")

# items = soup.select(".api_txt_lines.total_tit")

# for e, item in enumerate(items, 1):
#     print(f"{e} : {item.text}")

items = soup.select(".total_wrap.api_ani_send")

for rank_num, item in enumerate(items, 1):
    print(f"<<{rank_num}>>")
    ad = item.select_one(".link_ad")
    if ad:
        print("광고입니다.")
        continue

    blog_title = item.select_one(".sub_txt.sub_name").text
    print(f"{blog_title}")

    post_title = item.select_one(".api_txt_lines.total_tit._cross_trigger")
    print(f"{post_title.text}")

    print(f"{post_title.get('href')}")
    print(f"{post_title['href']}")

    print()

driver.quit()

# ------------------csv 파일 양식으로 만드는 부분 : Part_CSV----------------------------