import os
import time
import random
import re
import json
from collections import Counter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
from datetime import datetime


# ====== NLTK 불용어 적용 ======
import nltk
from nltk.corpus import stopwords as nltk_stopwords
nltk.download('stopwords')

# 기본 영어 불용어
stopwords = set(nltk_stopwords.words("english"))

# 논문/논문사이트 특화 불용어 추가
extra_stopwords = {
    "introduction", "method", "result", "figure", "table",
    "dataset", "experiment", "paper", "approach", "related", "work"
}
stopwords.update(extra_stopwords)

# ====== 설정 ======
base_year = 2025
start_week = 45
wait_time = 7
max_retry_per_article = 4
retry_click = 6

# ====== 로깅 설정 ======
# ====== 로깅 파일 이름 생성 ======
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_week_str = f"{base_year}-W{start_week:02d}"
log_file = f"crawling_{log_week_str}_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info(f"🚀 크롤링 시작 — 로그파일: {log_file}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("🚀 크롤링 시작")

# ====== 웹 드라이버 실행 ======
options = webdriver.ChromeOptions()
# options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("user-agent=Mozilla/5.0")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ====== 초기 주차 URL ======
week = start_week
week_url = f"https://huggingface.co/papers/week/{base_year}-W{week:02d}"
file_index = int(str(base_year)[-2:] + f"{week:02d}" + "001")

# ====== 통계 ======
total_articles = 0
success_count = 0
fail_count = 0

# ====== 크롤링 루프 ======
while True:
    logging.info(f"🔹 Crawling week URL: {week_url}")
    folder = f"{base_year}-W{week:02d}"
    os.makedirs(folder, exist_ok=True)
    time.sleep(random.uniform(3, 6))

    # 아티클 링크 추출
    try:
        driver.get(week_url)
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article h3 a"))
        )
        articles = driver.find_elements(By.CSS_SELECTOR, "article h3 a")
        article_urls = [a.get_attribute("href") for a in articles]
        logging.info(f"📝 {len(article_urls)} articles found")
    except Exception as e:
        logging.error(f"❌ No articles found or page error: {e}")
        break

    total_articles += len(article_urls)

    # 각 아티클 크롤링
    for link in article_urls:
        article_success = False
        for attempt in range(1, max_retry_per_article + 1):
            try:
                driver.get(link)
                time.sleep(random.uniform(3, 6))

                # 제목
                try:
                    paper_name = WebDriverWait(driver, wait_time).until(
                        EC.presence_of_element_located((By.TAG_NAME, "h1"))
                    ).text.strip()
                except:
                    paper_name = "Unknown_Title"

                # Abstract
                try:
                    abstract_div = WebDriverWait(driver, wait_time).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.pb-8.pr-4.md\\:pr-16 > div")
                        )
                    )
                    ps = abstract_div.find_elements(By.TAG_NAME, "p")
                    page_content = "\n".join([p.text.strip() for p in ps]) if ps else abstract_div.text.strip()
                except:
                    page_content = ""

                # Upvote
                try:
                    upvote_elem = WebDriverWait(driver, wait_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR,
                            "section.pt-8 div.hidden.flex-wrap.items-start.gap-2.md\\:flex a div div"
                        ))
                    )
                    upvote_match = re.search(r"\d+", upvote_elem.text.strip())
                    upvote = int(upvote_match.group()) if upvote_match else 0
                except:
                    upvote = 0

                # GitHub 링크
                try:
                    github_url = driver.find_element(By.XPATH, "//a[contains(@href,'github.com')]").get_attribute("href")
                except:
                    github_url = ""

                huggingface_url = link

                # 태그 추출
                words = re.findall(r'\b\w+\b', page_content.lower())
                filtered = [w for w in words if w not in stopwords and len(w) > 2]
                counter = Counter(filtered)
                tags = [tag for tag, _ in counter.most_common(3)]
                while len(tags) < 3:
                    tags.append("")

                # JSON 구조
                json_data = {
                    "content": page_content,
                    "metadata": {
                        "paper_name": paper_name,
                        "github_url": github_url,
                        "huggingface_url": huggingface_url,
                        "upvote": upvote,
                        "tags": tags
                    }
                }

                # 저장
                doc_name = f"doc{file_index}.json"
                file_path = os.path.join(folder, doc_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)

                logging.info(f"✅ Saved {file_path}")
                file_index += 1
                success_count += 1
                article_success = True
                break

            except Exception as e:
                logging.warning(f"⚠️ Retry {attempt}/{max_retry_per_article} failed for {link}, error: {e}")
                time.sleep(3)

        if not article_success:
            logging.error(f"❌ Failed to crawl article: {link}")
            fail_count += 1

    # ====== 다음 주 버튼 클릭 (XPath 사용) ======
    clicked = False
    for attempt in range(retry_click):
        try:
            driver.get(week_url)  # 주차 리스트 페이지로 이동
            next_btn = WebDriverWait(driver, wait_time).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/main/div[2]/section/div[1]/div[4]/div/div[2]/a[2]"))
            )
            next_btn.click()
            time.sleep(random.uniform(3, 6))
            week += 1
            week_url = driver.current_url
            file_index = int(str(base_year)[-2:] + f"{week:02d}" + "001")
            clicked = True
            logging.info(f"➡ Moving to next week: {week_url}")
            break
        except:
            logging.warning(f"⚠️ Next button click attempt {attempt+1}/{retry_click} failed")
            time.sleep(2)

    if not clicked:
        logging.info("➡ No more weeks. Crawling finished.")
        break

driver.quit()

# ====== 최종 통계 ======
logging.info("🎉 크롤링 완료!")
logging.info(f"총 아티클 수: {total_articles}")
logging.info(f"성공: {success_count}")
logging.info(f"실패: {fail_count}")
