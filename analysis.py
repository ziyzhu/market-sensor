import pyspark
from pyspark import SparkContext
from pyspark.sql import Row, SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import udf 
import os 
from datetime import timedelta
import stanza
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from tqdm import tqdm
from instrument import * 
from article import *

class AnalyticEngine:
    def __init__(self, symbols, startdate, enddate, interval, sc, data_dir='./data'):
        self.symbols = symbols
        self.startdate = startdate
        self.enddate = enddate
        self.interval = interval
        self.instruments = list()
        self.histories = list()
        self.data = dict()
        self.data_dir = data_dir
        self.spark = SparkSession(sc)

    def simulate(self, startdate, enddate):
        for k, v in self.data.items():
            price_series = v['timeline_df']['open']
            title_sentiment_series = v['timeline_df']['title_score']
            text_sentiment_series = v['timeline_df']['text_score']


    def add_score(self, window=3):

        def calc_score(index, article_df_col, article_df, window):
            average = lambda scores: int(sum(scores) / len(scores)) if len(scores) > 0 else None
            scores = []
            for day in range(window):
                article_links = None
                try:
                    article_links = timeline_df.loc[index - timedelta(days=day)]['links']
                except KeyError:
                    continue

                if not article_links:
                    continue

                for link in article_links:
                    try:
                        article = article_df.loc[link]
                    except KeyError:
                        continue
                    if article[article_df_col]:
                        scores.append(article[article_df_col])
            return average(scores)

        for k, v in tqdm(self.data.items()):
            article_df = v['article_df']
            timeline_df = v['timeline_df']
            if timeline_df.empty or article_df.empty:
                continue

            timeline_df['title_score'] = timeline_df.index.map(lambda index: calc_score(index, 'title_sentiment_score', article_df, window))
            timeline_df['text_score'] = timeline_df.index.map(lambda index: calc_score(index, 'text_sentiment_score', article_df, window))


    def add_sentiment(self):
        nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

        @udf
        def sentiment_score(s):
            if s:
                doc = nlp(s)
                sentiments = [s.sentiment for s in doc.sentences]
                score = int(50 * sum(sentiments) / len(sentiments))
                return score
        
        for k, v in tqdm(self.data.items()):
            article_df = v['article_df']
            if article_df.empty:
                continue
            
            df = self.spark.createDataFrame(article_df)
            df = df.repartition(30)
            df.cache()

            df = df.withColumn('title_sentiment_score', sentiment_score(df['title']))
            df = df.withColumn('text_sentiment_score', sentiment_score(df['text']))

            self.data[k]['article_df'] = df.toPandas()
            # sentiment_table[k] = df.select(['source', 'title_sentiment_score', 'text_sentiment_score']).collect()
            df.unpersist()


    def sample_article_dfs(self, save=True):
        article_dfs = [v['article_df'].sample(n=10) for v in self.data.values()]
        sample_df = pd.concat(article_dfs)
        if save:
            dest = f'./data/sample.csv'
            sample_df.to_csv(dest)
            print(f'saving sampled article dataframes to "{dest}"')
        return sample_df
        
    def load_all(self):
        self.load_instruments()
        self.load_histories(self.instruments)
        self.load_data()

    def init_data(self):
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
            timeline_df = timeline_df.rename(columns={'Date': 'date', 'Open': 'open'})
            timeline_df.index = pd.to_datetime(timeline_df.index)

            timeline_df.name = instrument.id
            article_df.name = instrument.id
    
            self.data[instrument.id] = dict()
            self.data[instrument.id]['timeline_df'] = timeline_df
            self.data[instrument.id]['article_df'] = article_df


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

    def load_data(self):

        def convert_links(s):
            return [link.strip("'") for link in s[1:-1].split(',')]

        df_fnames = [fname for fname in os.listdir(self.data_dir) if 'csv' in fname and AnalyticEngine.__name__ in fname]
        if len(df_fnames) == 0:
            print('failed to load data from cache, loading data from scratch')
            return self.init_data()

        keys = [fname.split('_')[1] for fname in df_fnames]
        for k in keys:
            article_df = None
            sentiment_df = None
            timeline_df = None
            for fname in df_fnames:
                if k in fname and 'sentiment_df' in fname:
                    sentiment_df = pd.read_csv(f'{self.data_dir}/{fname}', index_col='link')
                if k in fname and 'article_df' in fname:
                    article_df = pd.read_csv(f'{self.data_dir}/{fname}', index_col='link')
                    article_df = article_df.replace({np.nan: None})
                if k in fname and 'timeline_df' in fname:
                    timeline_df = pd.read_csv(f'{self.data_dir}/{fname}', index_col='date', converters={'links': lambda s: convert_links(s)})
                    timeline_df.index = pd.to_datetime(timeline_df.index)
                    timeline_df = timeline_df.replace({np.nan: None})

            self.data[k] = dict()
            self.data[k]['article_df'] = article_df
            self.data[k]['timeline_df'] = timeline_df
            self.data[k]['sentiment_df'] = sentiment_df

    def readcache_data(self):
        for objname in cache.listcache(f'{AnalyticEngine.__name__}'):
            symbol, df_name = objname.split('-')[1:]
            if symbol not in self.data:
                self.data[symbol] = dict()
            self.data[symbol][df_name] = cache.readcache(objname)

    def cache_data(self):
        for symbol, df_dict in tqdm(self.data.items()):
            for df_name, df in df_dict.items():
                cache.writecache(f'{AnalyticEngine.__name__}-{symbol}-{df_name}', df)
        
    def __repr__(self):
        return f'AnalyticEngine(data={len(self.data)}, histories={len(self.histories)})'

