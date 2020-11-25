import pyspark
from pyspark import SparkContext
from pyspark.sql import Row, SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import udf 
import stanza
import numpy as np
import pandas as pd 
from tqdm import tqdm
from instrument import * 
from article import *

class AnalyticEngine:
    def __init__(self, symbols, startdate, enddate, interval, sc):
        self.symbols = symbols
        self.startdate = startdate
        self.enddate = enddate
        self.interval = interval
        self.instruments = list()
        self.histories = list()
        self.data = dict()
        self.spark = SparkSession(sc)

    def simulate(self, startdate, enddate):
        pass
    
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
            if v['article_df'].empty:
                continue
            
            df = self.spark.createDataFrame(v['article_df'])
            df = df.repartition(30)
            df.cache()

            df = df.withColumn('title_sentiment_score', sentiment_score(df['title']))
            df = df.withColumn('text_sentiment_score', sentiment_score(df['text']))

            self.data[k]['article_df'] = df.toPandas()
            df.unpersist()

    def sample_article_dfs(self, save=True):
        sample_df = None
        article_dfs = [v['article_df'] for v in self.data.values()]
        subsample_dfs = [df.sample(n=10) for df in article_dfs]
        for df in subsample_dfs:
            if sample_df == None:
                sample_df = df
            sample_df = sample_df.unionByName(df)
        if save:
            dest = f'./data/sample.csv'
            sample_df.to_csv(dest)
            print(f'saving sampled article dataframes to "{dest}"')
        return sample_df
        
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
            timeline_df = timeline_df[['Open', 'links']].replace({np.nan: None}).reset_index(drop=False)
            timeline_df = timeline_df.rename(columns={'Date': 'date', 'Open': 'open'})

            timeline_df.name = instrument.id
            article_df.name = instrument.id
    
            self.data[instrument.id] = {'timeline_df': timeline_df,\
                                        'article_df': article_df}


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

    def save_data(self):
        for k, v in tqdm(self.data.items()):
            v['article_df'].to_csv(f'./data/{AnalyticEngine.__name__}_{k}_article_df.csv')
            v['timeline_df'].to_csv(f'./data/{AnalyticEngine.__name__}_{k}_timeline_df.csv')

    def load_data(self):
        pass
        
    def __repr__(self):
        return f'AnalyticEngine(data={len(self.data)}, histories={len(self.histories)})'

