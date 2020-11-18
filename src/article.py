from pygooglenews import GoogleNews
import newspaper
from tqdm import tqdm
from datetime import datetime, timedelta, date
from time import sleep
import random 
import requests, json
import cache
from stock import *

HEADERS = None
with open('../headers.json', 'r') as f:
    HEADERS = json.load(f)
    if not HEADERS:
        message = '''Missing Rapid API credentials'''
        print(message)

def formatdt(dt):
    if dt:
        return dt.strftime('%Y-%m-%d')
    return ''

class ArticleHistory:
    def __init__(self, instrument_id, startdate = None, enddate = None):
        self.instrument_id = instrument_id
        self.startdate = startdate
        self.enddate = enddate
        self.history = []
    
    def add(self, group):
        self.history.append(group)
    
    def load_text(self):
        for group in tqdm(self.history):
            group.load_text()

    def get_aligned_articles(self):
        aligned = []
        for group in self.history:
            articles = group.get_articles(ordered=True)
            aligned.extend(articles)
        return aligned
    
    def cache(self):
        startdate_str = formatdt(self.startdate)
        enddate_str = formatdt(self.enddate)
        cache_dict = self.__dict__.copy()
        cache_dict['history'] = [group.__dict__ for group in cache_dict['history']]
        cache.writecache(f'{self.__class__.__name__}_{self.instrument_id}_{startdate_str}_{enddate_str}', cache_dict)
        
    def readcache(self):
        query = f'{self.__class__.__name__}_{self.instrument_id}'
        for objname in cache.listcache(query):
            old_history_dict = cache.readcache(objname)
            self.startdate = datetime.strptime(objname.split('_')[2], '%Y-%m-%d')
            self.enddate = datetime.strptime(objname.split('_')[3], '%Y-%m-%d')
            for i, group_dict in enumerate(old_history_dict['history']):
                group = ArticleGroup(self.instrument_id)
                group.__dict__.update(group_dict)
                old_history_dict['history'][i] = group
            self.__dict__.update(old_history_dict)
            return True
        return False

    def __repr__(self):
        startdate_str = formatdt(self.startdate)
        enddate_str = formatdt(self.enddate)
        return f'ArticleHistory(instrument_id={self.instrument_id}, startdate={startdate_str}, enddate={enddate_str}, history={len(self.history)})'

class ArticleGroup:
    def __init__(self, instrument_id, articles = None, startdate = None, enddate = None):
        self.instrument_id = instrument_id
        self.articles = articles
        self.startdate = startdate
        self.enddate = enddate
        self.textloaded = False
    
    def load_text(self, refresh=False):

        if self.textloaded and not refresh:
            return

        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15'
        config = newspaper.Config()
        config.browser_user_agent = user_agent
        config.fetch_images = False
        config.memoize_articles = False 

        np_articles = []
        for article in self.articles['articles']:
            np_article = newspaper.Article(article['link'])
            np_articles.append(np_article)

        newspaper.news_pool.set(np_articles, threads_per_source=1)
        newspaper.news_pool.join()

        for np_article, article in zip(np_articles, self.articles['articles']):
            try:
                np_article.parse()
            except:
                continue
            article['text'] = np_article.text
        
        self.textloaded = True

    def get_articles(self, ordered=True):
        parsed = []
        for article in self.articles['articles']:
            published = datetime.strptime(article['published'], '%a, %d %b %Y %H:%M:%S %Z')
            article['published'] = published.replace(hour=0, minute=0, second=0, microsecond=0)
            parsed.append(article)
        parsed.sort(key=lambda k: k['published'])
        return parsed

    def __repr__(self):
        startdate_str = formatdt(self.startdate)
        enddate_str = formatdt(self.enddate)
        return f'ArticleGroup(instrument_id={self.instrument_id}, startdate={startdate_str}, enddate={enddate_str})'

def gn_search(q, fromtime, totime):
    sleep(random.uniform(0.4, 0.6))
    url = "https://google-news.p.rapidapi.com/v1/search"
    headers = HEADERS
    qstring = {"q": q, "from": fromtime, "to": totime, "country": "US", "lang": "en"}
    response = requests.get(url, headers=headers, params=qstring)
    articles = json.loads(response.text)
    return articles

def load_history(instrument, startdate, enddate, interval, readcache=True, writecache=False):
    history = ArticleHistory(instrument.id, startdate, startdate)
    if readcache:
        history.readcache()
        return history

    print(history)
    
    while history.enddate < enddate:
        try:
            nextdate = history.enddate + interval
            articles = gn_search(instrument.qstr(), formatdt(history.enddate), formatdt(nextdate))
            group = ArticleGroup(instrument.id, articles, history.enddate, nextdate)
            print(group)
        except Exception as e:
            print(e)
            sleep(1)
            continue
        history.add(group)
        history.enddate += interval

    if writecache:
        history.cache()

    return history

def load_histories(startdate, enddate, interval, readcache=True, writecache=False):

    instruments = load_instruments(startdate, enddate, readcache=True, writecache=False)
    histories = []

    for instrument in tqdm(instruments):
        history = load_history(instrument, startdate, enddate, interval, readcache, writecache)
        histories.push(history)

    return histories


