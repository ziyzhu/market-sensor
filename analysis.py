import numpy as np
import pandas as pd 
from instrument import * 
from article import *

class AnalyticEngine:
    def __init__(self, symbols, startdate, enddate, interval):
        self.symbols = symbols
        self.startdate = startdate
        self.enddate = enddate
        self.interval = interval
        self.instruments = list()
        self.histories = list()
        self.data = dict()

    def load_instruments(self):
        self.instruments = Instrument.load_instruments(self.symbols, self.startdate, self.enddate)

    def load_histories(self, instruments, download_article_content=False):
        for instrument in instruments:
            startdate, enddate = instrument.date_range()
            if not all([startdate, enddate]):
                continue

            history = ArticleHistory.load_history(instrument, startdate, enddate, self.interval)
            if download_article_content:
                history.load_text()

            self.histories.append(history)

    def prepare(self):
        self.load_instruments()
        self.load_histories(self.instruments)

        for instrument, history in zip(self.instruments, self.histories):
            articles = history.get_aligned_articles()
            articles_dict = dict()
            columns = ['published', 'title', 'link', 'source', 'id', 'text']
            for col in columns:
                articles_dict[col] = []
                for article in articles:
                    articles_dict[col].append(article.get(col))

            article_df = pd.DataFrame.from_dict(articles_dict).set_index('link', drop=False, verify_integrity=True)
            article_series = article_df.groupby('published')['link'].apply(list).rename('links')
            timeline_df = pd.merge(instrument.df, article_series, left_index=True, right_index=True, how='left')
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

