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
    symbols = [symbol for symbol_list in symbol_map.values() for symbol in symbol_list]

    startdate = datetime(2015, 1, 1)
    enddate = datetime(2017, 1, 1)
    interval = timedelta(weeks=4)

    sc = SparkContext("local[*]", "AnalyticEngine")
    engine = AnalyticEngine(symbol_map, startdate, enddate, interval, sc)
    engine.load_all()

    engine.analyze_cov(symbols, window=3, info='for all symbols')

    # for tag in symbol_map:
    #     engine.analyze_cov(symbol_map[tag], window=3, info=f'for {tag}')
    
    article_dfs = [df_dict['article_df'] for df_dict in engine.data.values()]
