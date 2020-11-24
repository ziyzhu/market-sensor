import numpy as np
import pandas as pd 
from instrument import * 
from article import *

class AnalyticEngine:
    def __init__(self, startdate, enddate, interval):
        self.startdate = startdate
        self.enddate = enddate
        self.interval = interval
        self.histories = list()
        self.data = dict()

    def prepare(self, download_article_content=False):
        # instruments = load_instruments(self.startdate, self.enddate, readcache=False, writecache=True)
        instruments = load_instruments(self.startdate, self.enddate, readcache=True, writecache=False)

        for instrument in instruments:
            instrument_df = instrument.df.copy()
            startdate, enddate = instrument.date_range()
            if not all([startdate, enddate]):
                continue

            # history = ArticleHistory.load_history(instrument, startdate, enddate, self.interval, readcache=False, writecache=True)
            history = ArticleHistory.load_history(instrument, startdate, enddate, self.interval, readcache=True, writecache=False)
            if download_article_content:
                history.load_text()
            self.histories.append(history)

            articles = history.get_aligned_articles()
            articles_dict = dict()
            columns = ['published', 'title', 'link', 'source', 'id', 'text']
            for col in columns:
                articles_dict[col] = []
                for article in articles:
                    articles_dict[col].append(article.get(col))

            article_df = pd.DataFrame.from_dict(articles_dict).set_index('link', drop=False, verify_integrity=True)
            article_series = article_df.groupby('published')['link'].apply(list).rename('links')
            timeline_df = pd.merge(instrument_df, article_series, left_index=True, right_index=True, how='left')
            timeline_df = timeline_df[['Open', 'links']].replace({np.nan: None})

            timeline_df.name = instrument.id
            article_df.name = instrument.id
    
            self.data[instrument.id] = {'timeline_df': timeline_df, 'article_df': article_df}

    def to_csv(self):
        for instrument_id, dataitem in self.data.items():
            dataitem['timeline_df'].to_csv(f'./data/{instrument_id}_timeline_df.csv')
            dataitem['article_df'].to_csv(f'./data/{instrument_id}_article_df.csv')
        
    def __repr__(self):
        return f'AnalyticEngine(data={len(self.data)}, histories={len(self.histories)})'

