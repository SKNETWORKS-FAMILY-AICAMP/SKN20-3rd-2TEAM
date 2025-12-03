"""
HuggingFace Weekly Papers Crawler (Selenium 버전)

크롤링 파이프라인:
1. https://huggingface.co/papers/week/2025-W46 페이지 접속
2. 페이지에서 논문 목록 로드 (동적 렌더링 대기)
3. 각 논문 카드 클릭하여 상세 페이지 접속
4. Abstract, 제목, GitHub URL, Upvote 추출
5. Abstract에서 키워드 3개 추출
6. doc{YY}{ww}{NNN}.json 형식으로 개별 저장
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
from pathlib import Path
from typing import List, Dict
import re
from collections import Counter
import time


class HFWeeklyCrawler:
    """HuggingFace Weekly Papers 크롤러 (Selenium)"""
    
    def __init__(self, base_dir: str = "SKN20-3rd-2TEAM/01_data/documents", headless: bool = True):
        """
        Args:
            base_dir: JSON 파일을 저장할 최상위 디렉토리
            headless: True면 브라우저 창을 표시하지 않음 (백그라운드 실행)
        """
        self.base_dir = Path(base_dir)
        self.headless = headless
        self.driver = None
        
    def init_driver(self):
        """Selenium WebDriver 초기화"""
        options = webdriver.ChromeOptions()
        
        if self.headless:
            options.add_argument('--headless')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
        print("✅ Selenium WebDriver 초기화 완료")
    
    def close_driver(self):
        """WebDriver 종료"""
        if self.driver:
            self.driver.quit()
            print("✅ WebDriver 종료")
    
    def parse_week_url(self, url: str) -> tuple:
        """URL에서 연도와 주차 추출
        
        Args:
            url: https://huggingface.co/papers/week/2025-W46 형식
            
        Returns:
            (year, week): (2025, 46)
        """
        pattern = r'/week/(\d{4})-W(\d{2})'
        match = re.search(pattern, url)
        if not match:
            raise ValueError(f"잘못된 URL 형식입니다: {url}\n올바른 형식: https://huggingface.co/papers/week/YYYY-WNN")
        
        year = int(match.group(1))
        week = int(match.group(2))
        return year, week
    
    def extract_keywords(self, abstract: str, top_n: int = 3) -> List[str]:
        """Abstract에서 상위 키워드 3개 추출 (TF 기반)
        
        Args:
            abstract: 논문 초록
            top_n: 추출할 키워드 개수 (기본값: 3)
            
        Returns:
            상위 키워드 리스트
        """
        if not abstract:
            return []
        
        # 불용어 제거
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'we', 'our', 'their', 'they', 'it', 'its', 'which', 'who', 'when',
            'where', 'why', 'how', 'what', 'if', 'than', 'such', 'into', 'through',
            'paper', 'propose', 'present', 'show', 'demonstrate', 'using', 'used',
            'approach', 'method', 'model', 'based', 'results', 'work'
        }
        
        # 단어 추출 및 필터링
        words = re.findall(r'\b[a-z]{3,}\b', abstract.lower())
        filtered_words = [w for w in words if w not in stopwords]
        
        # 빈도 계산 및 상위 N개 추출
        word_freq = Counter(filtered_words)
        keywords = [word for word, _ in word_freq.most_common(top_n)]
        
        return keywords
    
    def get_paper_urls(self, week_url: str) -> List[str]:
        """주차 페이지에서 모든 논문 URL 추출
        
        Args:
            week_url: https://huggingface.co/papers/week/2025-W46
            
        Returns:
            논문 상세 페이지 URL 리스트
        """
        print(f"🔄 페이지 로딩 중: {week_url}")
        self.driver.get(week_url)
        
        # 페이지 로딩 대기
        time.sleep(3)
        
        # 페이지를 스크롤하면서 모든 논문 카드 로드
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # 논문 링크 추출 (community 제외)
        paper_urls = []
        
        try:
            # 방법 1: article 태그 내의 링크
            articles = self.driver.find_elements(By.CSS_SELECTOR, "article a[href*='/papers/']")
            for article in articles:
                href = article.get_attribute('href')
                if href and '/papers/' in href and href not in paper_urls:
                    # /papers/week/, #community 경로 제외
                    if '/papers/week/' not in href and '#community' not in href:
                        paper_urls.append(href)
        except NoSuchElementException:
            pass
        
        try:
            # 방법 2: 직접 /papers/ 링크 찾기
            links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/papers/2']")
            for link in links:
                href = link.get_attribute('href')
                if href and href not in paper_urls:
                    # /papers/week/, #community 제외, 논문 ID 형식 확인
                    if '/papers/week/' not in href and '#community' not in href and re.search(r'/papers/\d{4}\.\d+', href):
                        paper_urls.append(href)
        except NoSuchElementException:
            pass
        
        # 중복 제거 및 #community가 포함된 URL 한번 더 필터링
        paper_urls = [url for url in dict.fromkeys(paper_urls) if '#community' not in url]
        
        print(f"✅ {len(paper_urls)}개 논문 URL 발견")
        return paper_urls
    
    def extract_paper_info(self, paper_url: str) -> Dict:
        """논문 상세 페이지에서 정보 추출
        
        Args:
            paper_url: 논문 상세 페이지 URL
            
        Returns:
            {
                "context": "Abstract",
                "metadata": {
                    "paper_name": "제목",
                    "github_url": "...",
                    "huggingface_url": "...",
                    "upvote": 123,
                    "tags": ["k1", "k2", "k3"]
                }
            }
        """
        self.driver.get(paper_url)
        time.sleep(2)
        
        paper_data = {
            "context": "",
            "metadata": {
                "paper_name": "",
                "github_url": "",
                "huggingface_url": paper_url,
                "upvote": 0,
                "tags": []
            }
        }
        
        try:
            # 제목 추출
            try:
                title_elem = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
                )
                paper_data["metadata"]["paper_name"] = title_elem.text.strip()
            except TimeoutException:
                paper_data["metadata"]["paper_name"] = "Unknown Title"
            
            # Abstract 추출 (여러 선택자 시도)
            abstract = ""
            selectors = [
                # HuggingFace Papers 페이지의 일반적인 구조
                "div.pb-8.pr-4.md\\:pr-16",  # 메인 컨텐츠 영역
                "div[class*='prose']",
                "div.prose",
                "article div",
                "main div.text-lg",
                "div[class*='abstract']",
                "div[class*='Abstract']",
                "section[class*='abstract']",
                "p[class*='abstract']"
            ]
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.text.strip()
                        # Abstract는 보통 길이가 100자 이상
                        if text and len(text) > 100:
                            # "Abstract" 단어가 포함되어 있거나 충분히 긴 텍스트
                            if 'abstract' in text.lower()[:100] or len(text) > 200:
                                abstract = text
                                # "Abstract" 헤더 제거
                                abstract = re.sub(r'^abstract\s*:?\s*', '', abstract, flags=re.IGNORECASE)
                                break
                    if abstract:
                        break
                except NoSuchElementException:
                    continue
            
            if not abstract:
                print(f"    ⚠️  Abstract를 찾을 수 없음 (페이지 구조 확인 필요)")
            else:
                print(f"    📄 Abstract 길이: {len(abstract)} 자")
            
            paper_data["context"] = abstract
            
            # GitHub URL 추출 (없으면 빈 문자열)
            github_url = ""
            try:
                github_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='github.com']")
                if github_links:
                    github_url = github_links[0].get_attribute('href')
            except NoSuchElementException:
                pass
            
            paper_data["metadata"]["github_url"] = github_url
            
            # Upvote 추출
            upvote = 0
            try:
                # 정확한 선택자: <div class="font-semibold text-orange-500">117</div>
                upvote_elem = self.driver.find_element(By.CSS_SELECTOR, "body > div > main > div > section.pt-8.border-gray-100.md\:col-span-5.pt-6.lg\:pt-28.pb-24.md\:pl-6.md\:border-l > div.hidden.flex-wrap.items-start.gap-2.md\:flex > div > div > a > div > div")
                upvote_text = upvote_elem.text.strip()
                # 숫자만 추출
                numbers = re.findall(r'\d+', upvote_text)
                if numbers:
                    upvote = int(numbers[0])
                    print(f"    ⭐ Upvote: {upvote}")
            except (NoSuchElementException, ValueError) as e:
                print(f"    ⚠️  Upvote를 찾을 수 없음")
            
            paper_data["metadata"]["upvote"] = upvote
            
            # 키워드 추출
            if abstract:
                paper_data["metadata"]["tags"] = self.extract_keywords(abstract, top_n=3)
            
        except Exception as e:
            print(f"    ⚠️  정보 추출 중 오류: {e}")
        
        return paper_data
    
    def save_individual_papers(self, papers_data: List[Dict], year: int, week: int):
        """각 논문을 개별 JSON 파일로 저장
        
        파일 형식: doc{YY}{ww}{NNN}.json
        예: doc2546001.json (2025년 46주차 1번)
        
        Args:
            papers_data: 변환된 논문 데이터 리스트
            year: 연도
            week: 주차
        """
        # 출력 디렉토리: SKN20-3rd-2TEAM/01_data/documents/2025/2025-W46
        year_dir = self.base_dir / str(year)
        week_dir = year_dir / f"{year}-W{week:02d}"
        week_dir.mkdir(parents=True, exist_ok=True)
        
        # YY: 연도 마지막 2자리 (2025 -> 25)
        yy = str(year)[-2:]
        
        print(f"\n💾 JSON 파일 저장 중...")
        for idx, paper in enumerate(papers_data, 1):
            # 파일명: doc{YY}{ww}{NNN}.json
            filename = f"doc{yy}{week:02d}{idx:03d}.json"
            filepath = week_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(paper, f, indent=2, ensure_ascii=False)
            
            title = paper['metadata']['paper_name'][:50]
            print(f"  ✅ [{idx}/{len(papers_data)}] {filename} - {title}...")
        
        print(f"\n✅ 총 {len(papers_data)}개 논문 저장 완료: {week_dir}")
    
    def crawl_week(self, week_url: str):
        """주차 URL의 논문을 크롤링하고 JSON 파일로 저장
        
        전체 파이프라인:
        1. URL에서 year, week 추출
        2. 주차 페이지에서 모든 논문 URL 수집
        3. 각 논문 페이지를 방문하여 정보 추출
        4. doc{YY}{ww}{NNN}.json 형식으로 저장
        
        Args:
            week_url: HuggingFace weekly papers URL
                     예: https://huggingface.co/papers/week/2025-W46
        """
        print("\n" + "="*70)
        print("📚 HuggingFace Weekly Papers Crawler (Selenium)")
        print("="*70)
        
        # STEP 1: URL 파싱
        print(f"\n🔗 입력 URL: {week_url}")
        try:
            year, week = self.parse_week_url(week_url)
            print(f"📅 크롤링 대상: {year}년 {week}주차 ({year}-W{week:02d})")
        except ValueError as e:
            print(f"❌ {e}")
            return
        
        # WebDriver 초기화
        self.init_driver()
        
        try:
            # STEP 2: 논문 URL 목록 수집
            print(f"\n{'─'*70}")
            print("STEP 1: 논문 URL 목록 수집")
            print(f"{'─'*70}")
            paper_urls = self.get_paper_urls(week_url)
            
            if not paper_urls:
                print("⚠️  논문 URL을 찾을 수 없습니다.")
                return
            
            # STEP 3: 각 논문 정보 추출
            print(f"\n{'─'*70}")
            print("STEP 2: 각 논문 정보 크롤링")
            print(f"{'─'*70}")
            papers_data = []
            
            for i, url in enumerate(paper_urls, 1):
                print(f"\n  [{i}/{len(paper_urls)}] 크롤링 중: {url}")
                try:
                    paper_info = self.extract_paper_info(url)
                    
                    # Abstract 유무와 관계없이 모두 저장
                    papers_data.append(paper_info)
                    title = paper_info['metadata']['paper_name'][:50]
                    tags = ', '.join(paper_info['metadata']['tags']) if paper_info['metadata']['tags'] else '없음'
                    print(f"    ✅ {title}... (태그: {tags})")
                    
                    time.sleep(1)  # 서버 부하 방지
                    
                except Exception as e:
                    print(f"    ❌ 오류 발생: {e}")
                    continue
            
            if not papers_data:
                print("❌ 추출된 논문이 없습니다.")
                return
            
            # STEP 4: 파일 저장
            print(f"\n{'─'*70}")
            print("STEP 3: JSON 파일 저장")
            print(f"{'─'*70}")
            self.save_individual_papers(papers_data, year, week)
            
            # 요약 통계
            print(f"\n{'='*70}")
            print("📊 크롤링 완료 요약")
            print(f"{'='*70}")
            print(f"  📁 저장 경로: {self.base_dir / str(year) / f'{year}-W{week:02d}'}")
            print(f"  📄 총 논문 수: {len(papers_data)}")
            print(f"  ⭐ 평균 Upvote: {sum(p['metadata']['upvote'] for p in papers_data) / len(papers_data):.1f}")
            print(f"  🔗 GitHub URL 포함: {sum(1 for p in papers_data if p['metadata']['github_url'])}")
            print(f"  📝 파일 형식: doc{str(year)[-2:]}{week:02d}001.json ~ doc{str(year)[-2:]}{week:02d}{len(papers_data):03d}.json")
            print("="*70 + "\n")
            
        finally:
            # WebDriver 종료
            self.close_driver()


def main():
    """메인 실행 함수"""
    # 크롤러 초기화 (headless=False로 설정하면 브라우저 창이 보임)
    crawler = HFWeeklyCrawler(
        base_dir="01_data/documents/2025/2025-W46",
        headless=True  # False로 변경하면 브라우저 창이 표시됨
    )
    
    # 단일 주차 크롤링
    crawler.crawl_week("https://huggingface.co/papers/week/2025-W46")
    
    # 여러 주차를 크롤링하려면:
    # urls = [
    #     "https://huggingface.co/papers/week/2025-W44",
    #     "https://huggingface.co/papers/week/2025-W45",
    #     "https://huggingface.co/papers/week/2025-W46",
    # ]
    # for url in urls:
    #     crawler.crawl_week(url)
    #     time.sleep(2)  # 서버 부하 방지


if __name__ == "__main__":
    main()