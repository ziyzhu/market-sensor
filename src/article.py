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
    def __init__(self, instrument_id=None, startdate=None, enddate=None):
        self.instrument_id = instrument_id
        self.startdate = startdate
        self.enddate = enddate
        self.groups = []
    
    def add(self, group):
        self.groups.append(group)
    
    def load_text(self):
        fails = 0
        for group in tqdm(self.groups):
            try:
                group.load_text(refresh=False)
            except Exception as e:
                print(e)
                fails += 1
        return fails

    def get_aligned_articles(self):
        aligned = []
        for group in self.groups:
            articles = group.get_articles(ordered=True)
            aligned.extend(articles)
        return aligned
    
    def cache(self):
        startdate_str = formatdt(self.startdate)
        enddate_str = formatdt(self.enddate)
        cache.writecache(f'{self.__class__.__name__}_{self.instrument_id}_{startdate_str}_{enddate_str}', self.to_dict())

    @staticmethod
    def readcache(instrument_id):
        query = f'{ArticleHistory.__name__}_{instrument_id}'
        objname = cache.findcache(query)
        if not objname:
            return None

        old_history_dict = cache.readcache(objname)
        return ArticleHistory.from_dict(old_history_dict)
    
    def to_dict(self):
        d = self.__dict__.copy()
        d['groups'] = [group.to_dict() for group in d['groups']]
        return d
    
    @staticmethod
    def from_dict(d):
        history = ArticleHistory()
        d.update({'groups': [ArticleGroup.from_dict(group_d) for group_d in d['groups']]})
        history.__dict__.update(d)
        return history

    def __repr__(self):
        startdate_str = formatdt(self.startdate)
        enddate_str = formatdt(self.enddate)
        return f'ArticleHistory(instrument_id={self.instrument_id}, startdate={startdate_str}, enddate={enddate_str}, groups={len(self.groups)})'

class ArticleGroup:
    def __init__(self, instrument_id=None, search=None, startdate=None, enddate=None):
        self.instrument_id = instrument_id
        self.search = search
        self.startdate = startdate
        self.enddate = enddate
    
    def load_text(self, refresh=False):

        if not refresh:
            return 

        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15'
        config = newspaper.Config()
        config.browser_user_agent = user_agent
        config.fetch_images = False
        config.memoize_articles = False 

        np_articles = []
        for article in self.search['articles']:
            np_article = newspaper.Article(article['link'])
            np_articles.append(np_article)

        newspaper.news_pool.set(np_articles, threads_per_source=1)
        newspaper.news_pool.join()

        for np_article, article in zip(np_articles, self.search['articles']):
            try:
                np_article.parse()
            except:
                continue
            article['text'] = np_article.text
            article['keywords'] = np_article.keywords
            article['summary'] = np_article.summary
        
        total = len(self.search['articles'])
        return 
        
    def get_articles(self, ordered=True):
        parsed = []
        for article in self.search['articles']:
            published = datetime.strptime(article['published'], '%a, %d %b %Y %H:%M:%S %Z')
            article['published'] = published.replace(hour=0, minute=0, second=0, microsecond=0)
            parsed.append(article)
        parsed.sort(key=lambda k: k['published'])
        return parsed
    
    @staticmethod
    def from_dict(d):
        group = ArticleGroup()
        group.__dict__.update(d)
        return group
    
    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        startdate_str = formatdt(self.startdate)
        enddate_str = formatdt(self.enddate)
        n_articles = len(self.search['articles'])
        return f'ArticleGroup(instrument_id={self.instrument_id}, startdate={startdate_str}, enddate={enddate_str}, search={n_articles})'

def gn_search(q, fromtime, totime):
    sleep(random.uniform(0.4, 0.6))
    url = "https://google-news.p.rapidapi.com/v1/search"
    headers = HEADERS
    qstring = {"q": q, "from": fromtime, "to": totime, "country": "US", "lang": "en"}
    response = requests.get(url, headers=headers, params=qstring)
    search = json.loads(response.text)
    return search

def load_history(instrument, startdate, enddate, interval, readcache=True, writecache=False):
    if readcache:
        history = ArticleHistory.readcache(instrument.id)
        return history

    history = ArticleHistory(instrument.id, startdate, startdate)
    while history.enddate < enddate:
        try:
            nextdate = history.enddate + interval
            search = gn_search(instrument.qstr(), formatdt(history.enddate), formatdt(nextdate))
            group = ArticleGroup(instrument.id, search, history.enddate, nextdate)
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


