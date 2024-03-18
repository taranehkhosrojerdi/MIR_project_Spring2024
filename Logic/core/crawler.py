import requests
from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import re


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = []
        self.crawled = []
        self.added_ids = set()
        self.crawled_lock = Lock()
        self.not_crawled_lock = Lock()
        self.added_ids_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        parts = URL.split('/')
        title_index = parts.index('title')
        movie_id = parts[title_index + 1]
        movie_id = movie_id.split('?')[0]
        return movie_id

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # crawled_list = list(self.crawled)
        # with open('IMDB_crawled.json', 'r') as f:
        #     file_data = json.load(f)

        with open('../IMDB_crawled.json', 'w') as f:
            json.dump(self.crawled, f, indent=4) 

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = json.load(f)

        with open('IMDB_not_crawled.json', 'r') as f:
            self.not_crawled = json.load(f)

        if self.not_crawled is not None:
            for obj in self.crawled:
                if obj in self.not_crawled:
                    self.not_crawled.remove(obj)

        with open('IMDB_not_crawled.json', 'w') as f:
            json.dump(self.not_crawled, f)

        for obj in self.crawled:
            self.added_ids.add(obj['id'])
        for link in self.not_crawled:
            self.added_ids.add(link[27:-1])

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        response = requests.get(URL, headers=self.headers)
        return response

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        response = requests.get(self.top_250_URL, headers=self.headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            regex = re.compile(r'/(tt\d{4,8})/')
            movie_ids = regex.findall(str(soup))
            for id in movie_ids:
                url = 'https://imdb.com/title/' + id
                self.not_crawled_lock.acquire()
                self.not_crawled.append(url)
                self.not_crawled_lock.release()

                self.added_ids_lock.acquire()
                self.added_ids.add(id)
                self.added_ids_lock.release()

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter < self.crawling_threshold:
                self.not_crawled_lock.acquire()
                URL = self.not_crawled.pop()
                self.not_crawled_lock.release()

                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_counter += 1
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        new_instance = self.get_imdb_instance()
        new_instance['id'] = self.get_id_from_URL(URL)

        self.extract_movie_info(new_instance, URL)
        self.added_ids_lock.acquire()
        for link in new_instance['related_links']:
            related_id = link.split('/')[-1]

            self.not_crawled_lock.acquire()
            if related_id not in self.added_ids:
                self.not_crawled.append(link)
            self.not_crawled_lock.release()
        self.added_ids_lock.release()

        self.crawled_lock.acquire()
        if new_instance not in self.crawled:
            self.crawled.append(new_instance)
        self.crawled_lock.release()

    def extract_movie_info(self, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        URL = URL + '/'
        res = requests.get(URL, headers=IMDbCrawler.headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        parents_url = URL + 'parentalguide'
        parents_res = requests.get(parents_url, headers=IMDbCrawler.headers)
        parents_soup = BeautifulSoup(parents_res.content, 'html.parser')

        summary_url = IMDbCrawler.get_summary_link(URL)
        summary_res = requests.get(summary_url, headers=IMDbCrawler.headers)
        summary_soup = BeautifulSoup(summary_res.content, 'html.parser')

        review_url = IMDbCrawler.get_review_link(URL)
        review_res = requests.get(review_url, headers=IMDbCrawler.headers)
        review_soup = BeautifulSoup(review_res.content, 'html.parser')

        movie['title'] = IMDbCrawler.get_title(soup)
        movie['first_page_summary'] = IMDbCrawler.get_first_page_summary(soup)
        movie['release_year'] = IMDbCrawler.get_release_year(soup)
        movie['mpaa'] = IMDbCrawler.get_mpaa(parents_soup)
        movie['budget'] = IMDbCrawler.get_budget(soup)
        movie['gross_worldwide'] = IMDbCrawler.get_gross_worldwide(soup)
        movie['directors'] = IMDbCrawler.get_director(soup)
        movie['writers'] = IMDbCrawler.get_writers(soup)
        movie['stars'] = IMDbCrawler.get_stars(soup)
        movie['related_links'] = IMDbCrawler.get_related_links(soup)
        movie['genres'] = IMDbCrawler.get_genres(soup)
        movie['languages'] = IMDbCrawler.get_languages(soup)
        movie['countries_of_origin'] = IMDbCrawler.get_countries_of_origin(soup)
        movie['rating'] = IMDbCrawler.get_rating(soup)
        movie['summaries'] = IMDbCrawler.get_summary(summary_soup)
        movie['synopsis'] = IMDbCrawler.get_synopsis(summary_soup)
        movie['reviews'] = IMDbCrawler.get_reviews_with_scores(review_soup)


    def get_summary_link(url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            summary_link = url + "plotsummary"
            return summary_link
        except Exception as e:
            print("failed to get summary link:", e)

    def get_review_link(url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            summary_link = url + "reviews"
            return summary_link
        except Exception as e:
            print("failed to get summary link:", e)

    def get_title(soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            section = soup.find('section', {'class': 'ipc-page-section ipc-page-section--baseAlt ipc-page-section--tp-none ipc-page-section--bp-xs sc-491663c0-2 eGWcuq'})
            title_h1 = section.find('h1', {'data-testid': 'hero__pageTitle'})
            title_div = title_h1.find('span', {'data-testid': 'hero__primary-text'})
            if title_div:
                return title_div.text.strip()
            else:
                return "Title not found"
        except Exception as e:
            print("failed to get title:", e)

    def get_first_page_summary(soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            section = soup.find('section', {'class': 'ipc-page-section ipc-page-section--baseAlt ipc-page-section--tp-none ipc-page-section--bp-xs sc-491663c0-2 eGWcuq'})
            inner_section = section.find('section', {'class': 'sc-67fa2588-4 gflGWU'})
            summary_tag = inner_section.find('p', {'data-testid': 'plot'})
            summary = summary_tag.find('span', {'data-testid': 'plot-xs_to_m'})
            if summary:
                return summary.text.strip()
            else:
                return "Summary not found"
        except Exception as e:
            print("failed to get first page summary:", e)

    def get_director(soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            script_tag = soup.find('script', type='application/ld+json')
            if script_tag:
                json_data = json.loads(script_tag.string)
                directors = [director['name'] for director in json_data.get('director', [])]
                if directors:
                    return directors
                else:
                    return "Directors not found"
            else:
                return "Script tag not found"
        except Exception as e:
            print("Failed to get directors:", e)

    def get_stars(soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            script_tag = soup.find('script', type='application/ld+json')
            if script_tag:
                json_data = json.loads(script_tag.string)
                stars = [actor['name'] for actor in json_data.get('actor', [])]
                if stars:
                    return stars
                else:
                    return "Stars not found"
            else:
                return "Script tag not found"
        except Exception as e:
            print("Failed to get stars:", e)

    def get_writers(soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            script_tag = soup.find('script', type='application/ld+json')
            if script_tag:
                json_data = json.loads(script_tag.string)
                writers = [writer['name'] for writer in json_data.get('creator', []) if writer.get('@type') == 'Person']
                if writers:
                    return writers
                else:
                    return "Writers not found"
            else:
                return "Script tag not found"
        except Exception as e:
            print("Failed to get writers:", e)

    def get_related_links(soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            related = []
            layer1 = soup.find_all('div',
                                        class_='ipc-poster ipc-poster--base ipc-poster--dynamic-width ipc-poster-card__poster ipc-sub-grid-item ipc-sub-grid-item--span-2')

            for div in layer1:
                layer2 = div.find('a', class_='ipc-lockup-overlay').get('href')
                regex = re.compile(r'/title/(tt\d{6,8})/')
                found_strings = regex.search(layer2)
                related.append(found_strings.group())
            related = ['https://www.imdb.com' + r for r in related]
            return related
        except:
            print("failed to get related links")
            return ['N/A']

    def get_summary(soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            summaries = []
            summary_sections = soup.find_all('script', id='__NEXT_DATA__', type='application/json')
            for section in summary_sections:
                summary_data = json.loads(section.string)
                if 'props' in summary_data and 'pageProps' in summary_data['props'] \
                        and 'contentData' in summary_data['props']['pageProps'] \
                        and 'categories' in summary_data['props']['pageProps']['contentData']:
                    categories = summary_data['props']['pageProps']['contentData']['categories']
                    for category in categories:
                        if category['id'] == 'summaries':
                            for item in category['section']['items']:
                                summary = item['htmlContent'].strip()
                                summary = re.sub(r'<span.*', '', summary)
                                summaries.append(summary)
            return summaries if summaries else ["Summary not found"]
        except Exception as e:
            print("Failed to get summary:", e)

    def get_synopsis(soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            synopsis_script = soup.find('script', id='__NEXT_DATA__', type='application/json')
            if synopsis_script:
                synopsis_data = json.loads(synopsis_script.string)
                synopsis_section = None
                for category in synopsis_data['props']['pageProps']['contentData']['categories']:
                    if category['name'] == 'Synopsis':
                        synopsis_section = category['section']
                        break
                if synopsis_section:
                    synopsis_html = synopsis_section['items'][0]['htmlContent']
                    synopsis_soup = BeautifulSoup(synopsis_html, 'html.parser')
                    synopsis_text = synopsis_soup.get_text(separator='\n', strip=True)
                    return [synopsis_text]
                else:
                    return ["Synopsis not found"]
            else:
                return ["Synopsis not found"]
        except Exception as e:
            print("Failed to get synopsis:", e)

    def get_reviews_with_scores(soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            reviews_with_scores = []
            review_containers = soup.find_all('div', class_='content')
            for review_container in review_containers:
                review_text = review_container.find('div', class_='text show-more__control').text.strip()
                striped_text = review_text.split('.')
                score = striped_text[len(striped_text) - 2]
                score = score[1:len(score) - 3]
                reviews_with_scores.append([review_text, score])

            return reviews_with_scores
        except Exception as e:
            print("failed to get reviews:", e)

    def get_genres(soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            script_tags = soup.find_all('script', type='application/ld+json')
            for tag in script_tags:
                json_data = json.loads(tag.string)
                if 'genre' in json_data:
                    return json_data['genre']
            return []
        except Exception as e:
            print("Failed to get genres:", e)
            return []

    def get_rating(soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            script_tags = soup.find_all('script', type='application/ld+json')
            for tag in script_tags:
                json_data = json.loads(tag.string)
                if 'aggregateRating' in json_data:
                    return str(json_data['aggregateRating']['ratingValue'])
            return "Rating not found"
        except Exception as e:
            print("Failed to get rating:", e)
            return "Rating not found"

    def get_mpaa(soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            layer0 = soup.find('section', {'class': 'article listo content-advisories-index'})
            layer1 = layer0.find('section', {'id': 'certificates'})
            layer2 = layer1.find('table', {'class': 'ipl-zebra-list row'})
            layer4 = layer2.find('tr', {'id': 'mpaa-rating'})
            if layer4:
                rating = layer4.find_all('td')[1]
                rating = rating.text.strip()
                return rating.split()[1]
            else:
                return "Rating Not Found"
        except Exception as e:
            print("Failed to get MPAA title:", e)

    def get_release_year(soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.text.strip()
                year_start_index = title_text.find("(") + 1
                year_end_index = title_text.find(")")
                release_year = title_text[year_start_index:year_end_index]
                return release_year
            else:
                return "Release year not found"
        except Exception as e:
            print("Failed to get release year:", e)
            return "Release year not found"

    def get_languages(soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            details = soup.find('section', {'data-testid': "Details"})
            details_li = details.find('li', {'data-testid': "title-details-languages"})
            details_div = details_li.find('div', {'class': 'ipc-metadata-list-item__content-container'})
            details_ul = details_div.find('ul', {'class': 'ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content base'})
            languages = []
            final_details = details_ul.find_all('li', {'class': 'ipc-inline-list__item'})
            for fd in final_details:
                languages.append(fd.text.strip())
            return languages
        except Exception as e:
            print("Failed to get languages:", e)
            return None

    def get_countries_of_origin(soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            countries_container = soup.find('span', class_='ipc-metadata-list-item__label',
                                            string='Countries of origin')
            if not countries_container:
                countries_container = soup.find('span', class_='ipc-metadata-list-item__label',
                                                string='Country of origin')
            if countries_container:
                country_links = countries_container.find_next('ul').find_all('a',
                                                                             class_='ipc-metadata-list-item__list-content-item--link')
                countries = [link.text.strip() for link in country_links]
                return countries
            else:
                return ["Countries of origin not found"]
        except Exception as e:
            print("Failed to get countries of origin:", e)
            return None


    def get_budget(soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            budget_container = soup.find('section', {'data-testid': 'BoxOffice'})
            if budget_container:
                budget_container = budget_container.find('div', {'data-testid': 'title-boxoffice-section'})
                if budget_container:
                    budget_text = budget_container.find('li', {'data-testid': 'title-boxoffice-budget'}).find(
                        'span', {'class': 'ipc-metadata-list-item__list-content-item'})
                    return budget_text.text.strip()
                else:
                    return "Budget not found"
            else:
                return "Budget not found"
        except Exception as e:
            print("Failed to get budget:", e)
            return "Failed to get budget"

    def get_gross_worldwide(soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            gross_container = soup.find('section', {'data-testid': 'BoxOffice'})
            gross_container = gross_container.find('li', {'data-testid': "title-boxoffice-cumulativeworldwidegross"})
            if gross_container:
                gross_container = gross_container.find('div', {'class': 'ipc-metadata-list-item__content-container'})
                gross_li = gross_container.find('li', {'class': 'ipc-inline-list__item'})
                gross_text = gross_li.find('span', {'class': 'ipc-metadata-list-item__list-content-item'})
                return gross_text.text.strip()
            else:
                return "Gross Worldwide Not Found"

        except Exception as e:
            print("Failed to get gross worldwide amount:", e)
            return "Failed to get gross worldwide amount"


def main():
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    # }
    #
    # url_bank = ["https://www.imdb.com/title/tt0111161/", "https://www.imdb.com/title/tt0099348/", "https://www.imdb.com/title/tt1454029/", "https://www.imdb.com/title/tt0053198/"]
    # for url in url_bank:
    #     response = requests.get(url, headers=headers)
    #     soup = BeautifulSoup(response.content, 'html.parser')
    #     title = IMDbCrawler.get_first_page_summary(soup)
    #     print(title)

    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
