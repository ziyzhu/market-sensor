from pyspark import SparkContext
import numpy as np
import pandas as pd 
from instrument import * 
from article import *
from analysis import *
from stats import * 

if __name__ == '__main__':

    symbol_map = {'semiconductor': ['QCOM', 'ASML', 'INTC', 'NVDA'],\
                  'automobile': ['TM', 'WKHS', 'GM', 'TSLA', 'FCAU', 'CVX'],\
                  'oil': ['BP', 'XOM', 'TOT'],\
                  'airline': ['AAL', 'LUV', 'DAL', 'UAL', 'JBLU']}
    startdate = datetime(2015, 1, 1)
    enddate = datetime(2017, 1, 1)
    interval = timedelta(weeks=4)

    sc = SparkContext("local[*]", "AnalyticEngine")
    engine = AnalyticEngine(symbol_map, startdate, enddate, interval, sc)
    engine.load_all()
    engine.analyze()

    # article_df = engine.data['QCOM']['article_df']
    # timeline_df = engine.data['QCOM']['timeline_df']
    # sentiment_df = engine.data['QCOM']['sentiment_df']
    # sample_df = engine.sample_article_dfs()


