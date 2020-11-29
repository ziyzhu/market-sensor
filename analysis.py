import os, math
from collections import Counter
from datetime import timedelta
import stanza
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from tqdm import tqdm
import dask.dataframe as dd
from dask.multiprocessing import get

from instrument import * 
from article import *

class AnalyticEngine:
    def __init__(self, symbol_map, startdate, enddate, interval, data_dir='./data'):
        self.symbol_map = symbol_map
        self.startdate = startdate
        self.enddate = enddate
        self.interval = interval
        self.instruments = list()
        self.histories = list()
        self.data = dict()
        self.data_dir = data_dir
        self.source_df = None
    
    def graph(self, symbols, window=3):
        for symbol in symbols:
            self.add_score_and_accuracy(window=window)
            fig, axes = plt.subplots(2)
            df_dict = self.data[symbol]
            price_series = df_dict['timeline_df']['open'].plot.line(ax=axes[0])
            title_sentiment_series = df_dict['timeline_df']['title_score'].plot.line(ax=axes[1])
            text_sentiment_series = df_dict['timeline_df']['text_score'].plot.line(ax=axes[1])
            plt.legend()
            plt.show()

    def analyze_accuracy(self, symbols, window=3, info='', save_fig=False, show_fig=False):
        res = dict()
        self.add_score_and_accuracy(window=window)
        for symbol in symbols:
            res[symbol] = dict()
            timeline_df = self.data[symbol]['timeline_df']
            title_correct = timeline_df[timeline_df['title_accuracy'] > 0]['title_accuracy'].count();
            text_correct = timeline_df[timeline_df['text_accuracy'] > 0]['text_accuracy'].count();
            res[symbol]['title_accuracy'] = title_correct / len(timeline_df)
            res[symbol]['text_accuracy'] = text_correct / len(timeline_df)

        plt.bar(symbols, [res[symbol]['title_accuracy'] for symbol in symbols], label='title_accuracy')
        plt.bar(symbols, [res[symbol]['text_accuracy'] for symbol in symbols], label='text_accuracy')
        plt.ylabel('accuracy % / 100')
        plt.legend()
        if save_fig:
            plt.savefig(f'./chart/accuracy_{window}.jpg')
        if show_fig:
            plt.show()
        plt.clf()
        return res

    def analyze_accuracies(self, symbols, windows, save_fig=False, show_fig=False):
        res = dict()
        for window in windows:
            subres = self.analyze_accuracy(symbols, window=window, info=f'window={window}', show_fig=show_fig)
            res[window] = dict()
            res[window]['title_accuracy'] = sum([subres[symbol]['title_accuracy'] for symbol in subres]) / len(subres)
            res[window]['text_accuracy'] = sum([subres[symbol]['text_accuracy'] for symbol in subres]) / len(subres)

        plt.bar(windows, [res[window]['title_accuracy'] for window in windows], label='title_accuracy')
        plt.bar(windows, [res[window]['text_accuracy'] for window in windows], label='text_accuracy')
        plt.ylabel('accuracy % / 100')
        plt.xlabel('Window in days')
        plt.legend()
        if save_fig:
            plt.savefig(f'./chart/accuracies.jpg')
        if show_fig:
            plt.show()
        plt.clf()
        return res
    
    def analyze_cov(self, symbols, window=3, info='', save_fig=False, show_fig=False):
        res = dict()
        res['price_title'] = []
        res['price_text'] = []
        res['title_text'] = []
        self.add_score_and_accuracy(window=window)
        for symbol in symbols:
            df_dict = self.data[symbol]
            res['price_title'].append(df_dict['timeline_df']['change'].cov(df_dict['timeline_df']['title_score']))
            res['price_text'].append(df_dict['timeline_df']['change'].cov(df_dict['timeline_df']['text_score']))
            res['title_text'].append(df_dict['timeline_df']['title_score'].cov(df_dict['timeline_df']['text_score']))

        plt.bar(symbols, res['price_title'], label='price change and title_score covariance')
        plt.bar(symbols, res['price_text'], label='price change and text_score covariance')
        plt.bar(symbols, res['title_text'], label='title_score and text_score covariance')
        plt.title(f'Covariance Analysis: window={window}, info={info}')
        plt.ylabel('Covariance Normalized by N-1 (Unbiased Estimator)')
        plt.legend()
        if save_fig:
            plt.savefig(f'./chart/cov_{window}.jpg')
        if show_fig:
            plt.show()
        plt.clf()
        return res

    def analyze_covs(self, symbols, windows, save_fig=False, show_fig=False):
        res = dict()
        for window in windows:
            covs = self.analyze_cov(symbols, window=window, info='for all symbols', show_fig=show_fig)
            res[window] = dict()
            for k in covs:
                res[window][k] = sum(covs[k])
        
        for k in ['price_text', 'price_title', 'title_text']: 
            plt.plot(windows, [res[window][k] for window in windows], label=k)

        plt.ylabel(f'Covariance Sum for {len(windows)} window days')
        plt.xlabel('Window in Days')
        plt.legend()
        if save_fig:
            plt.savefig(f'./chart/covs.jpg')
        if show_fig:
            plt.show()
        plt.clf()
        return res

    def calc_score(self, currdate, article_df_col, article_df, timeline_df, window):
        average = lambda scores: sum(scores) / len(scores) if len(scores) > 0 else None
        n_articles = 0
        scores = []
        days = [window - 1 if window > 0 else 0, window, window + 1] # sliding window method
        for day in days:
            article_links = None
            try:
                article_links = timeline_df.loc[currdate - timedelta(days=day)]['links']
            except KeyError:
                continue

            if not article_links:
                continue

            for link in article_links:
                try:
                    article = article_df.loc[link]
                    sentiment = article[article_df_col]
                    source_url = article['source']['href']
                    score = sentiment 
                    score *= self.source_df.loc[source_url]['weight'] # not enough weight
                except:
                    continue
                scores.append(score)

            n_articles += len(article_links)

        final_score = average(scores) 
        # final_score *= math.log(len(article_links)) # to be verified
        return final_score

    
    def calc_accuracy(self, score, change):
        if (score > 50 and change > 0) or (score < 50 and change < 0):
            return 100
        else:
            return 0

    def add_score_and_accuracy(self, window=3):
        '''
        parallelize this function / use spark
        '''
        for symbol, df_dict in tqdm(self.data.items()):
            article_df = df_dict['article_df']
            timeline_df = df_dict['timeline_df']
            if all([timeline_df.empty, article_df.empty]):
                continue

            timeline_df['title_score'] = timeline_df.index.map(lambda index: self.calc_score(index, 'title_sentiment', article_df, timeline_df, window))
            timeline_df['text_score'] = timeline_df.index.map(lambda index: self.calc_score(index, 'text_sentiment', article_df, timeline_df, window))
            timeline_df['title_accuracy'] = timeline_df.apply(lambda row: self.calc_accuracy(row['title_score'], row['change']), axis=1)
            timeline_df['text_accuracy'] = timeline_df.apply(lambda row: self.calc_accuracy(row['text_score'], row['change']), axis=1)
    
    def sample_article_dfs(self, save=True):
        article_dfs = [df_dict['article_df'].sample(n=10) for df_dict in self.data.values()]
        sample_df = pd.concat(article_dfs)
        if save:
            dest = f'./data/sample.csv'
            sample_df.to_csv(dest)
            print(f'saving sampled article dataframes to "{dest}"')
        return sample_df
        
    def load_all(self):
        ''' 
        load cached dataframes
        '''

        self.load_instruments()
        self.load_histories(self.instruments)
        self.load_data(from_cache=True)
        self.load_source_df()

    def load_instruments(self):
        self.instruments = Instrument.load_instruments(self.symbol_map, self.startdate, self.enddate)

    def load_histories(self, instruments, download_article_content=False):
        '''
        load cached ArticleHistory objects
        '''

        for instrument in instruments:
            startdate, enddate = instrument.date_range()
            if not all([startdate, enddate]):
                continue

            history = ArticleHistory.load_history(instrument, startdate, enddate, self.interval)
            if download_article_content:
                history.download_text()

            self.histories.append(history)

    def load_source_df(self):
        '''
        creates a source-article-count dataframe 
        '''

        article_dfs = [df_dict['article_df'] for df_dict in self.data.values()]
        source_urls = list()
        for article_df in article_dfs:
            source_urls += article_df['source'].apply(lambda source: source['href']).tolist()
        source_count = Counter(source_urls)
        total = sum(source_count.values())
        source_dict = {url: ((count + total) / total) for url, count in source_count.items()}
        self.source_df = pd.DataFrame.from_dict(source_dict, orient='index', columns=['weight'])

    def load_data(self, from_cache=True):
        if from_cache:
            for objname in cache.listcache(f'{AnalyticEngine.__name__}'):
                symbol, df_name = objname.split('-')[1:]
                if symbol not in self.data:
                    self.data[symbol] = dict()
                self.data[symbol][df_name] = cache.readcache(objname)
        else:
            self.add_all()


    def cache_data(self):
        for symbol, df_dict in tqdm(self.data.items()):
            for df_name, df in df_dict.items():
                cache.writecache(f'{AnalyticEngine.__name__}-{symbol}-{df_name}', df)

    def add_all(self):
        ''' 
        create dataframes from cached objects
        '''

        self.load_instruments()
        self.load_histories(self.instruments)
        self.add_data()

    def add_data(self):
        ''' 
        creates dataframes from objects in cache
        '''

        for instrument, history in zip(self.instruments, self.histories):
            articles = history.get_aligned_articles()
            articles_dict = dict()
            columns = ['published', 'title', 'link', 'source', 'id', 'text', 'title_sentiment', 'text_sentiment']
            for col in columns:
                articles_dict[col] = []
                for article in articles:
                    articles_dict[col].append(article.get(col))

            article_df = pd.DataFrame.from_dict(articles_dict).set_index('link', drop=False, verify_integrity=True)
            article['published'] = pd.to_datetime(article_df['published'])
            article_series = article_df.groupby('published')['link'].apply(list).rename('links')
            timeline_df = pd.merge(instrument.df, article_series, left_index=True, right_index=True, how='left')
            timeline_df = timeline_df[['Open', 'Close', 'links']].replace({np.nan: None})
            timeline_df = timeline_df.rename(columns={'Date': 'date', 'Open': 'open', 'Close': 'close'})
            timeline_df['change'] = timeline_df.apply(lambda row: 100 * (row['close'] - row['open']) / row['close'], axis=1)
            timeline_df.index = pd.to_datetime(timeline_df.index)

            timeline_df.name = instrument.id
            article_df.name = instrument.id
    
            self.data[instrument.id] = dict()
            self.data[instrument.id]['timeline_df'] = timeline_df
            self.data[instrument.id]['article_df'] = article_df

    def __repr__(self):
        return f'AnalyticEngine(data={len(self.data)}, histories={len(self.histories)})'

