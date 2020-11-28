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
    def __init__(self, symbol_map, startdate, enddate, interval, sc, data_dir='./data'):
        self.symbol_map = symbol_map
        self.startdate = startdate
        self.enddate = enddate
        self.interval = interval
        self.instruments = list()
        self.histories = list()
        self.data = dict()
        self.data_dir = data_dir
        self.source_df = None
        self.spark = SparkSession(sc)
    
    def graph(self):
        for symbol, df_dict in self.data.items():
            price_series = df_dict['timeline_df']['open'].plot.line()
            title_sentiment_series = df_dict['timeline_df']['title_score'].plot.line()
            text_sentiment_series = df_dict['timeline_df']['text_score'].plot.line()
            plt.legend()
            plt.show()

    def analyze_cov(self, symbols, window=3, info='', save_fig=False, show_fig=True):
        covs = dict()
        covs['price_title'] = []
        covs['price_text'] = []
        covs['title_text'] = []
        self.score(window=window)
        for symbol in symbols:
            df_dict = self.data[symbol]
            covs['price_title'].append(df_dict['timeline_df']['open'].cov(df_dict['timeline_df']['title_score']))
            covs['price_text'].append(df_dict['timeline_df']['open'].cov(df_dict['timeline_df']['text_score']))
            covs['title_text'].append(df_dict['timeline_df']['title_score'].cov(df_dict['timeline_df']['text_score']))

        plt.bar(symbols, covs['price_title'], label='price and title_score covariance')
        plt.bar(symbols, covs['price_text'], label='price and text_score covariance')
        plt.bar(symbols, covs['title_text'], label='title_score and text_score covariance')
        plt.title(f'Covariance Analysis: window={window}, info={info}')
        plt.ylabel('Covariance Normalized by N-1 (Unbiased Estimator)')
        plt.xlabel('Instrument/Stock Symbol')
        plt.legend()
        if save_fig:
            plt.savefig(f'./chart/window{window}_covariance')
        if show_fig:
            plt.show()
        plt.clf()
        return covs

    def score(self, window=3):

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

        for symbol, df_dict in tqdm(self.data.items()):
            article_df = df_dict['article_df']
            timeline_df = df_dict['timeline_df']
            if timeline_df.empty or article_df.empty:
                continue

            timeline_df['title_score'] = timeline_df.index.map(lambda index: calc_score(index, 'title_sentiment_score', article_df, window))
            timeline_df['text_score'] = timeline_df.index.map(lambda index: calc_score(index, 'text_sentiment_score', article_df, window))
    
    def sample_article_dfs(self, save=True):
        article_dfs = [df_dict['article_df'].sample(n=10) for df_dict in self.data.values()]
        sample_df = pd.concat(article_dfs)
        if save:
            dest = f'./data/sample.csv'
            sample_df.to_csv(dest)
            print(f'saving sampled article dataframes to "{dest}"')
        return sample_df
        
    def load_all(self):
        self.load_instruments()
        self.load_histories(self.instruments)
        self.load_data(from_cache=True)

    def load_instruments(self):
        self.instruments = Instrument.load_instruments(self.symbol_map, self.startdate, self.enddate)

    def load_histories(self, instruments, download_article_content=False):
        for instrument in instruments:
            startdate, enddate = instrument.date_range()
            if not all([startdate, enddate]):
                continue

            history = ArticleHistory.load_history(instrument, startdate, enddate, self.interval)
            if download_article_content:
                history.load_text()

            self.histories.append(history)

    def load_data(self, from_cache=True):
        if from_cache:
            for objname in cache.listcache(f'{AnalyticEngine.__name__}'):
                symbol, df_name = objname.split('-')[1:]
                if symbol not in self.data:
                    self.data[symbol] = dict()
                self.data[symbol][df_name] = cache.readcache(objname)
        else:
            self.add_all()

    def load_source_df(self):
        source_dict = dict()
        article_dfs = [df_dict['article_df'] for df_dict in self.data.values()]
        for article_df in article_dfs:
            source_urls = article_df['source'].apply(lambda source: source['href']).tolist()
            for url in source_urls:
                source_dict[url] = None
        self.source_df = pd.DataFrame.from_dict(source_dict, orient='index', columns=['rating'])

    def cache_data(self):
        for symbol, df_dict in tqdm(self.data.items()):
            for df_name, df in df_dict.items():
                cache.writecache(f'{AnalyticEngine.__name__}-{symbol}-{df_name}', df)

    def add_all(self):
        self.add_data()
        self.add_sentiment()

    def add_data(self):
        ''' 
        creates dataframes from objects in cache
        '''

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


    def add_sentiment(self):
        ''' 
        offloads work to Spark to improve performance 
        '''

        nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

        @udf
        def sentiment_score(s):
            if s:
                doc = nlp(s)
                sentiments = [s.sentiment for s in doc.sentences]
                score = int(50 * sum(sentiments) / len(sentiments))
                return score
        
        for symbol, df_dict in tqdm(self.data.items()):
            article_df = df_dict['article_df']
            if article_df.empty:
                continue
            
            df = self.spark.createDataFrame(article_df)
            df = df.repartition(30)
            df.cache()

            df = df.withColumn('title_sentiment_score', sentiment_score(df['title']))
            df = df.withColumn('text_sentiment_score', sentiment_score(df['text']))

            if not self.data[symbol]:
                raise

            self.data[symbol]['article_df'] = df.select(['source', 'title_sentiment_score', 'text_sentiment_score']).toPandas()
            df.unpersist()
        
    def __repr__(self):
        return f'AnalyticEngine(data={len(self.data)}, histories={len(self.histories)})'

